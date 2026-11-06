"""Segmentação rastreável de relatórios e resumos em inglês."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class Sentence:
    id: str
    index: int
    text: str


@dataclass(frozen=True)
class Passage:
    id: str
    sentence_ids: tuple[str, ...]
    start_index: int
    end_index: int
    text: str


def split_sentences(text: str, prefix: str) -> list[Sentence]:
    """Retorna sentenças em ordem original, sem descartar conteúdo silenciosamente.

    A segmentação é feita por pySBD, que lida melhor com abreviaturas do
    inglês. O fallback regex permite testar o projeto antes de instalar deps;
    para pesquisa, instalar pySBD é recomendado.
    """
    try:
        import pysbd
        raw_parts = pysbd.Segmenter(language="en", clean=False).segment(text)
    except ImportError:
        raw_parts = re.split(r"(?<=[.!?])\s+(?=[\"'A-Z0-9(])", text.strip())
    results: list[Sentence] = []
    for part in raw_parts:
        normalized = re.sub(r"\s+", " ", part).strip()
        if normalized:
            index = len(results)
            results.append(Sentence(f"{prefix}_{index:04d}", index, normalized))
    return results


def build_passages(sentences: Iterable[Sentence], max_window: int = 2) -> list[Passage]:
    """Indexa sentenças isoladas e janelas contíguas, preservando IDs de origem."""
    items = list(sentences)
    if not 1 <= max_window <= 3:
        raise ValueError("max_window deve estar entre 1 e 3.")
    passages: list[Passage] = []
    for start in range(len(items)):
        for window_size in range(1, max_window + 1):
            window = items[start:start + window_size]
            if len(window) != window_size:
                break
            passages.append(Passage(
                id=f"P_{start:04d}_{start + window_size - 1:04d}",
                sentence_ids=tuple(s.id for s in window),
                start_index=start,
                end_index=start + window_size - 1,
                text=" ".join(s.text for s in window),
            ))
    return passages
