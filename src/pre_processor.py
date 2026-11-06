"""Prepara evidências rastreáveis no formato consumido pela interface HTML."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

# Compatível com execução direta e com importação como src.pre_processor.
if __package__:
    from .evidence.dataset_io import DATASET_NAME, load_govreport_records, load_local_records
    from .evidence.segmentation import build_passages, split_sentences
    from .evidence.translation import CachedTranslator, TranslationCache
else:
    from evidence.dataset_io import DATASET_NAME, load_govreport_records, load_local_records
    from evidence.segmentation import build_passages, split_sentences
    from evidence.translation import CachedTranslator, TranslationCache

ROOT = Path(__file__).resolve().parents[1]
BIENCODER = 'sentence-transformers/all-MiniLM-L6-v2'
RERANKER = 'cross-encoder/ms-marco-MiniLM-L6-v2'


class RetrievalProcessor:
    def __init__(self, device='cpu', pool_size=30, max_window=2, bi_encoder=BIENCODER,
                 cache_path=None, translate=True, retrieval_mode="hybrid", rrf_k=60):
        from sentence_transformers import SentenceTransformer, CrossEncoder
        self.device, self.pool_size, self.max_window = device, pool_size, max_window
        self.bi_encoder_name = bi_encoder
        self.retrieval_mode, self.rrf_k = retrieval_mode, rrf_k
        print('Carregando modelos de recuperação...', flush=True)
        self.bi_encoder = SentenceTransformer(bi_encoder, device=device)
        self.cross_encoder = CrossEncoder(RERANKER, device=device)
        self.translator = CachedTranslator(TranslationCache(cache_path or ROOT/'data/cache/translations.sqlite3'), device) if translate else None

    def process_document(self, doc_id, document_text, summary_text, top_k=5):
        if __package__:
            from .evidence.retrieval import EvidenceRetriever
        else:
            from evidence.retrieval import EvidenceRetriever
        if self.pool_size < top_k:
            raise ValueError('pool_size deve ser maior ou igual a top_k.')
        sentences = split_sentences(document_text, 'SRC')
        claims = split_sentences(summary_text, 'SUM')
        if not sentences or not claims:
            raise ValueError(f'{doc_id}: documento ou resumo sem sentenças utilizáveis')
        retriever = EvidenceRetriever(build_passages(sentences, self.max_window), self.bi_encoder, self.cross_encoder,
                                      retrieval_mode=self.retrieval_mode, rrf_k=self.rrf_k)
        results = []
        for claim in claims:
            candidates = retriever.retrieve(claim.text, top_k, self.pool_size)
            texts = [claim.text] + [c.passage.text for c in candidates]
            translations = self.translator.translate_many(texts) if self.translator else {}
            results.append({
                'doc_id': doc_id,
                'summary_sentence_id': f'{doc_id}_sum_{claim.index}',
                'summary_sentence_en': claim.text,
                'summary_sentence_pt': translations.get(claim.text, ''),
                'retrieved_top_k': [{
                    'doc_sent_id': c.passage.id,
                    'source_sentence_ids': list(c.passage.sentence_ids),
                    'start_index': c.passage.start_index,
                    'end_index': c.passage.end_index,
                    'text_en': c.passage.text,
                    'text_pt': translations.get(c.passage.text, ''),
                    'bi_encoder_score': c.cosine_similarity,
                    'cross_encoder_score': c.reranker_score,
                    'bm25_score': c.bm25_score,
                    'rrf_score': c.rrf_score,
                    'dense_rank': c.dense_rank,
                    'bm25_rank': c.bm25_rank,
                } for c in candidates],
                'retrieval_config': {'bi_encoder': self.bi_encoder_name, 'reranker': RERANKER,
                                     'pool_size': self.pool_size, 'max_window': self.max_window,
                                     'retrieval_mode': self.retrieval_mode, 'rrf_k': self.rrf_k,
                                     'bm25': {'variant': 'okapi-positive-idf', 'k1': 1.5, 'b': 0.75,
                                              'tokenizer': 'unicode-casefold-v1'} if self.retrieval_mode == 'hybrid' else None,
                                     'top_k': top_k, 'translation_model': self.translator.model if self.translator else None},
            })
            print(f'[{doc_id}] Sentença {claim.index + 1}/{len(claims)}', flush=True)
        return results


def prepare_records(records, processor, output_path, split='test', top_k=5, offset=0, dataset=DATASET_NAME):
    """Publica o JSONL completo somente após processar todos os documentos."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path, total = None, 0
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=output_path.parent,
                                suffix='.jsonl.tmp', delete=False) as f:
            temporary_path = Path(f.name)
            seen = set()
            for index, record in enumerate(records, start=offset):
                doc_id = str(record.get('id') or f'govreport_{split}_{index:05d}')
                if doc_id in seen:
                    raise ValueError(f'ID de documento duplicado: {doc_id}')
                seen.add(doc_id)
                items = processor.process_document(doc_id=doc_id, document_text=record['report'],
                                                   summary_text=record['summary'], top_k=top_k)
                for item in items:
                    item.update(dataset=dataset, split=split, source_offset=index)
                    f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
                    total += 1
        if not total:
            raise ValueError('Nenhuma sentença produzida; destino preservado.')
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(f'Concluído: {len(records)} documentos, {total} sentenças em {output_path}.')
    return total


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('o valor deve ser maior que zero')
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('o valor deve ser não-negativo')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=positive_int, default=5)
    parser.add_argument('--offset', type=nonnegative_int, default=0)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='test')
    parser.add_argument('--top-k', type=positive_int, default=5)
    parser.add_argument('--retrieval-mode', choices=['hybrid', 'dense'], default='hybrid')
    parser.add_argument('--rrf-k', type=positive_int, default=60)
    parser.add_argument('--pool-size', type=positive_int, default=30)
    parser.add_argument('--max-window', type=int, choices=[1, 2, 3], default=2)
    parser.add_argument('--bi-encoder', default=BIENCODER)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--input', type=Path, help='JSONL local com report, summary e id opcional')
    parser.add_argument('--cache', type=Path, default=ROOT/'data/cache/translations.sqlite3')
    parser.add_argument('--no-translation', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'data/processed/retrieval_ready.jsonl')
    parser.add_argument('--overwrite', action='store_true', help='Permitir substituir entrada existente')
    args = parser.parse_args()
    if args.pool_size < args.top_k:
        parser.error('--pool-size deve ser maior ou igual a --top-k')
    if args.output.exists() and not args.overwrite:
        parser.error('Destino já existe. Use outro --output ou --overwrite explicitamente.')
    if args.input and args.input.resolve() == args.output.resolve():
        parser.error('--input e --output devem ser arquivos diferentes')
    try:
        import torch
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        if device == 'cuda' and not torch.cuda.is_available():
            parser.error('CUDA indisponível. Use --device cpu.')
        records = (load_local_records(args.input, args.limit, args.offset) if args.input else
                   load_govreport_records(args.limit, args.split, args.offset))
        processor = RetrievalProcessor(device, args.pool_size, args.max_window, args.bi_encoder,
                                       args.cache, not args.no_translation, args.retrieval_mode, args.rrf_k)
        prepare_records(records, processor, args.output, args.split, args.top_k, args.offset,
                        'local' if args.input else DATASET_NAME)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Erro: {exc}\nVerifique dependências, memória e acesso aos modelos.\n')


if __name__ == '__main__':
    main()
