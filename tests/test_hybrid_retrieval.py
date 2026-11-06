import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evidence.bm25 import BM25Index, reciprocal_rank_fusion, tokenize


class BM25Tests(unittest.TestCase):
    def test_score_matches_okapi_formula(self):
        index = BM25Index(['alpha alpha beta', 'beta'])
        expected = math.log(2) * 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 3 / 2))
        self.assertAlmostEqual(index.scores('alpha')[0], expected)
        self.assertEqual(index.scores('alpha')[1], 0)
        self.assertEqual(index.scores('missing'), [0, 0])
        self.assertEqual(index.scores('alpha alpha')[0], 2 * index.scores('alpha')[0])

    def test_tokenization_empty_and_length_normalization(self):
        self.assertEqual(tokenize('EPA, 2024; $1.5; AB-12!'), ['epa', '2024', '1.5', 'ab-12'])
        self.assertEqual(BM25Index(['', '!!!']).scores('EPA'), [0, 0])
        self.assertEqual(BM25Index([]).scores('EPA'), [])
        scores = BM25Index(['EPA', 'EPA unrelated unrelated unrelated']).scores('epa')
        self.assertGreater(scores[0], scores[1])

    def test_rrf_uses_positions_and_merges_duplicates(self):
        scores = reciprocal_rank_fusion([[0, 1], [1, 2]], 60)
        self.assertAlmostEqual(scores[0], 1/61)
        self.assertAlmostEqual(scores[1], 1/62 + 1/61)
        self.assertAlmostEqual(scores[2], 1/62)
        self.assertGreater(scores[1], scores[0])
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[0]], 0)


try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'Instale numpy para testar a recuperação híbrida.')
class HybridTests(unittest.TestCase):
    def retriever(self, mode='hybrid'):
        from src.evidence.retrieval import EvidenceRetriever
        from src.evidence.segmentation import Passage
        texts = ['General report.', 'Generic analysis.', 'Other findings.', 'EPA AB-12 inspection.']
        class Encoder:
            def encode(self, texts, **kwargs):
                return np.array([[1., 0.]]) if len(texts) == 1 else np.array([[1., 0.], [.9, math.sqrt(.19)], [.5, math.sqrt(.75)], [0., 1.]])
        class Reranker:
            def __init__(self):
                self.pairs = []
            def predict(self, pairs, **kwargs):
                self.pairs = pairs
                return np.array([10. if 'AB-12' in text else 1. for _, text in pairs])
        self.reranker = Reranker()
        return EvidenceRetriever([Passage(f'P{i}', (f'SRC{i}',), i, i, text) for i, text in enumerate(texts)],
                                 Encoder(), self.reranker, retrieval_mode=mode)

    def test_bm25_rescues_candidate_missed_by_dense_pool(self):
        hybrid = self.retriever().retrieve('EPA AB-12', top_k=1, pool_size=2)
        self.assertEqual(hybrid[0].passage.id, 'P3')
        self.assertGreater(hybrid[0].bm25_score, 0)
        self.assertEqual(hybrid[0].bm25_rank, 1)
        self.assertIsNone(hybrid[0].dense_rank)
        self.assertAlmostEqual(hybrid[0].rrf_score, 1/61)
        self.assertEqual(len(self.reranker.pairs), 2)
        dense = self.retriever('dense').retrieve('EPA AB-12', top_k=1, pool_size=2)
        self.assertNotEqual(dense[0].passage.id, 'P3')
        self.assertIsNone(dense[0].rrf_score)

    def test_no_lexical_match_falls_back_to_dense_order(self):
        retriever = self.retriever()
        first = retriever.retrieve('unmatched', top_k=2, pool_size=2)
        second = retriever.retrieve('unmatched', top_k=2, pool_size=2)
        self.assertEqual(first, second)
        dense = self.retriever('dense').retrieve('unmatched', top_k=2, pool_size=2)
        self.assertEqual([c.passage.id for c in first], [c.passage.id for c in dense])
        self.assertTrue(all(c.bm25_score == 0 and c.bm25_rank is None for c in first))

    def test_invalid_retrieval_configuration(self):
        retriever = self.retriever()
        with self.assertRaises(ValueError):
            retriever.retrieve('EPA', top_k=3, pool_size=2)
        with self.assertRaises(ValueError):
            retriever.retrieve('EPA', pool_size=2.5)
        from src.evidence.retrieval import EvidenceRetriever
        with self.assertRaises(ValueError):
            EvidenceRetriever(retriever.passages, retriever.bi_encoder, retriever.cross_encoder, rrf_k=0)


if __name__ == '__main__':
    unittest.main()
