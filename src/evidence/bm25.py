"""Okapi BM25 local, indexado uma vez por documento, sem dependência externa."""
from collections import Counter, defaultdict
from math import log1p
import re


def tokenize(text):
    """Normaliza caixa; preserva números e termos com hífen/apóstrofo."""
    return re.findall(r"\w+(?:[.'’-]\w+)*", text.casefold())


class BM25Index:
    def __init__(self, texts, k1=1.5, b=0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError('Parâmetros BM25 inválidos.')
        self.k1, self.b = k1, b
        self.lengths = []
        self.postings = defaultdict(list)
        for index, text in enumerate(texts):
            terms = tokenize(text)
            self.lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                self.postings[term].append((index, frequency))
        self.size = len(self.lengths)
        self.average_length = sum(self.lengths) / self.size if self.size else 0
        self.idf = {term: log1p((self.size - len(rows) + 0.5) / (len(rows) + 0.5))
                    for term, rows in self.postings.items()}

    def scores(self, query):
        scores = [0.0] * self.size
        if not self.average_length:
            return scores
        for term, query_frequency in Counter(tokenize(query)).items():
            for index, frequency in self.postings.get(term, ()):
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                scores[index] += query_frequency * self.idf[term] * frequency * (self.k1 + 1) / (frequency + norm)
        return scores


def reciprocal_rank_fusion(rankings, constant=60):
    """Soma 1/(constant+posição), com posições iniciadas em 1."""
    if constant < 1:
        raise ValueError('A constante RRF deve ser positiva.')
    scores = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for rank, index in enumerate(ranking, 1):
            if index not in seen:
                scores[int(index)] += 1 / (constant + rank)
                seen.add(index)
    return dict(scores)
