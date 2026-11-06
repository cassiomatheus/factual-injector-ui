import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest('Instale numpy para os testes de recuperação.')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evidence.segmentation import split_sentences, build_passages
from src.evidence.retrieval import EvidenceRetriever
from src.evidence.translation import TranslationCache, CachedTranslator, LONG_TEXT
from src.evidence.dataset_io import load_local_records, load_govreport_records
from src.pre_processor import RetrievalProcessor, prepare_records
from test_retrieval_html import ui, fixture, review


class Encoder:
    def encode(self, texts, **kwargs):
        return np.array([[1., 0.]] * len(texts))


class Reranker:
    def predict(self, pairs, **kwargs):
        # Prefere janela 0-1, depois sentença 1 (suprimida), depois sentença 2.
        return np.array([1., 5., 4., 3., 2.][:len(pairs)])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_short_sentences_windows_and_overlap(self):
        sentences = split_sentences('Yes. No. End.', 'SRC')
        self.assertEqual([s.text for s in sentences], ['Yes.', 'No.', 'End.'])
        passages = build_passages(sentences, 2)
        retriever = EvidenceRetriever(passages, Encoder(), Reranker())
        candidates = retriever.retrieve('Question?', top_k=3, pool_size=30)
        ids = [sid for c in candidates for sid in c.passage.sentence_ids]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(candidates[0].passage.sentence_ids, ('SRC_0000', 'SRC_0001'))
        self.assertEqual(candidates[1].passage.text, 'End.')

    def test_translation_cache_deduplicates_and_guards_long_input(self):
        cache = TranslationCache(self.root/'cache.sqlite3')
        translator = CachedTranslator(cache)
        calls = []
        class Pipeline:
            tokenizer = staticmethod(lambda text, **kw: SimpleNamespace(input_ids=[0] * (500 if text == 'long' else 2)))
            def __call__(self, batch, **kwargs):
                calls.append(batch)
                return [{'translation_text': 'PT:' + text} for text in batch]
        translator.pipeline = Pipeline()
        result = translator.translate_many(['Hello', 'Hello', 'long'])
        self.assertEqual(result, {'Hello': 'PT:Hello', 'long': LONG_TEXT})
        self.assertEqual(calls, [['Hello']])
        resumed = CachedTranslator(cache)
        self.assertEqual(resumed.translate_many(['Hello']), {'Hello': 'PT:Hello'})
        self.assertIsNone(resumed.pipeline)
        self.assertIsNone(cache.get('Hello', 'different-model'))

    def test_failed_translation_is_not_cached(self):
        translator = CachedTranslator(TranslationCache(self.root/'cache.sqlite3'))
        class Pipeline:
            tokenizer = staticmethod(lambda *a, **k: SimpleNamespace(input_ids=[1]))
            def __call__(self, *a, **k):
                raise RuntimeError('offline')
        translator.pipeline = Pipeline()
        with self.assertRaises(RuntimeError):
            translator.translate_many(['Hello'])
        self.assertIsNone(translator.cache.get('Hello', translator.model))

    def test_stream_offset_and_local_validation(self):
        rows = [{'report': f'Report {i}', 'summary': 'Summary.'} for i in range(4)]
        loader = SimpleNamespace(load_dataset=lambda *a, **k: iter(rows))
        with patch.dict(sys.modules, datasets=loader):
            self.assertEqual(load_govreport_records(2, 'test', 1), rows[1:3])
            with self.assertRaises(ValueError):
                load_govreport_records(5)
        source = self.root/'source.jsonl'
        source.write_text('\n'.join(json.dumps(row) for row in rows))
        self.assertEqual(load_local_records(source, 1, 2), rows[2:3])

    def test_processor_output_accepted_by_existing_ui(self):
        processor = RetrievalProcessor.__new__(RetrievalProcessor)
        processor.pool_size, processor.max_window = 30, 2
        processor.bi_encoder, processor.cross_encoder = Encoder(), Reranker()
        processor.bi_encoder_name, processor.translator = 'fake', None
        processor.retrieval_mode, processor.rrf_k = 'hybrid', 60
        rows = processor.process_document('d1', 'Yes. No. End.', 'Is that correct?', top_k=3)
        target = self.root/'ready.jsonl'
        target.write_text('\n'.join(json.dumps(row) for row in rows))
        items, identity = ui.load_data(target)
        self.assertEqual(items, rows)
        self.assertEqual(items[0]['retrieval_config']['retrieval_mode'], 'hybrid')
        self.assertEqual(items[0]['retrieval_config']['rrf_k'], 60)
        self.assertIn('bm25_score', items[0]['retrieved_top_k'][0])
        self.assertIn('rrf_score', items[0]['retrieved_top_k'][0])
        store = ui.ReviewStore(self.root/'reviews.sqlite3', items, identity)
        labels = {e['doc_sent_id']: ui.LABELS[0] for e in items[0]['retrieved_top_k']}
        saved = store.save_many({items[0]['summary_sentence_id']: review(status='completed', labels=labels)})
        self.assertEqual(next(iter(saved.values()))['status'], 'completed')

    def test_failed_preparation_preserves_existing_output(self):
        target = self.root/'ready.jsonl'
        target.write_text('original')
        processor = SimpleNamespace(process_document=lambda **kw: (_ for _ in ()).throw(RuntimeError('failed')))
        with self.assertRaises(RuntimeError):
            prepare_records([{'report': 'r', 'summary': 's'}], processor, target)
        self.assertEqual(target.read_text(), 'original')
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_history_migration_conflict_rollback_and_isolation(self):
        source = self.root/'input.jsonl'
        fixture(source)
        items, identity = ui.load_data(source)
        dbpath = self.root/'reviews.sqlite3'
        # Banco do formato anterior, sem tabela de histórico.
        with sqlite3.connect(dbpath) as db:
            db.execute('CREATE TABLE reviews(dataset TEXT, sid TEXT, version INTEGER, body TEXT, PRIMARY KEY(dataset,sid))')
            db.execute('INSERT INTO reviews VALUES (?,?,?,?)', (identity, 'doc1_sum_0', 1, json.dumps(review())))
        store = ui.ReviewStore(dbpath, items, identity)
        self.assertEqual(len(store.history()), 1)
        store.save_many({'doc1_sum_0': review(version=1)})
        with self.assertRaises(ui.ConflictError):
            store.save_many({'doc1_sum_1': review(), 'doc1_sum_0': review()})
        self.assertEqual([e['version'] for e in store.history()], [1, 2])
        self.assertNotIn('doc1_sum_1', store.snapshot())
        self.assertEqual(ui.ReviewStore(dbpath, items, 'other').history(), [])
        self.assertEqual(len(ui.ReviewStore(dbpath, items, identity).history()), 2)


if __name__ == '__main__':
    unittest.main()
