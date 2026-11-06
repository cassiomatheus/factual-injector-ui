"""BM25 + bi-encoder (cosseno) -> RRF -> cross-encoder -> Top-K diversificado."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Any
import numpy as np

from .segmentation import Passage
from .bm25 import BM25Index, reciprocal_rank_fusion


BIENCODER_NAME = "sentence-transformers/all-mpnet-base-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
RETRIEVAL_POOL = 30
MAX_WINDOW = 2


@dataclass(frozen=True)
class Candidate:
    rank: int
    passage: Passage
    cosine_similarity: float
    reranker_score: float
    bm25_score: float = 0.0
    rrf_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None


class EvidenceRetriever:
    """Indexação por documento; embeddings calculados uma única vez por documento.

    Passagem e consulta são textos em inglês. O ranking cross-encoder é de
    RELEVÂNCIA, não de implicação/contradição lógica.
    """
    def __init__(self, passages: Sequence[Passage], bi_encoder: Any, cross_encoder: Any,
                 retrieval_mode: str = "hybrid", rrf_k: int = 60):
        if not passages:
            raise ValueError("Nenhuma passagem para indexar.")
        if retrieval_mode not in ('hybrid', 'dense'):
            raise ValueError('retrieval_mode deve ser hybrid ou dense.')
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 1:
            raise ValueError('rrf_k deve ser inteiro positivo.')
        self.retrieval_mode, self.rrf_k = retrieval_mode, rrf_k
        self.passages = list(passages)
        self.bm25 = BM25Index([p.text for p in self.passages]) if retrieval_mode == 'hybrid' else None
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder
        self.embeddings = np.asarray(bi_encoder.encode(
            [p.text for p in self.passages],
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ))
        if self.embeddings.ndim != 2:
            raise RuntimeError("Embeddings com dimensões inesperadas.")

    def retrieve(self, query: str, top_k: int = 5, pool_size: int = RETRIEVAL_POOL) -> list[Candidate]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k deve ser inteiro positivo.")
        if isinstance(pool_size, bool) or not isinstance(pool_size, int) or pool_size < top_k:
            raise ValueError("pool_size não pode ser menor que top_k.")
        query_vec = np.asarray(self.bi_encoder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        ))[0]
        cosines = self.embeddings @ query_vec
        # A ordenação estável por índice facilita reproduzir empates.
        dense_indices = [int(i) for i in np.argsort(-cosines, kind="stable")[:pool_size]]
        dense_ranks = {index: rank for rank, index in enumerate(dense_indices, 1)}
        bm25_scores = self.bm25.scores(query) if self.bm25 is not None else [0.0] * len(self.passages)
        # Sem correspondência lexical não há contribuição artificial de scores zero.
        lexical_indices = sorted((i for i, score in enumerate(bm25_scores) if score > 0),
                                 key=lambda i: (-bm25_scores[i], i))[:pool_size]
        lexical_ranks = {index: rank for rank, index in enumerate(lexical_indices, 1)}
        fusion = reciprocal_rank_fusion([dense_indices, lexical_indices], self.rrf_k) if self.bm25 is not None else {}
        candidate_indices = (sorted(fusion, key=lambda i: (-fusion[i], -float(cosines[i]), i))[:pool_size]
                             if fusion else dense_indices)
        pairs = [[query, self.passages[int(idx)].text] for idx in candidate_indices]
        raw = np.asarray(self.cross_encoder.predict(pairs, show_progress_bar=False)).reshape(-1)
        if len(raw) != len(candidate_indices):
            raise RuntimeError("Reranker retornou número inválido de scores.")
        # Não interpretar logits do MS MARCO como probabilidades de factualidade.
        ordered = sorted(
            range(len(candidate_indices)),
            key=lambda i: (-float(raw[i]), -fusion.get(int(candidate_indices[i]), 0.0),
                           -float(cosines[int(candidate_indices[i])]), int(candidate_indices[i])),
        )
        results = []
        already_seen: set[str] = set()
        for pos in ordered:
            idx = int(candidate_indices[pos])
            passage = self.passages[idx]
            # Evita mostrar 1 sentença e sua janela adjacente como duas evidências
            # praticamente duplicadas no Top-K. O original permanece recuperável.
            if already_seen.intersection(passage.sentence_ids):
                continue
            results.append(Candidate(
                rank=len(results) + 1,
                passage=passage,
                cosine_similarity=float(cosines[idx]),
                reranker_score=float(raw[pos]),
                bm25_score=float(bm25_scores[idx]),
                rrf_score=fusion.get(idx),
                dense_rank=dense_ranks.get(idx),
                bm25_rank=lexical_ranks.get(idx),
            ))
            already_seen.update(passage.sentence_ids)
            if len(results) >= top_k:
                break
        return results
