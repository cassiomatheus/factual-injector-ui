"""Leitura validada de documentos, sem carregar modelos na importação."""
import json
from itertools import islice
from pathlib import Path

DATASET_NAME = 'ccdv/govreport-summarization'


def validate_records(records):
    if not records:
        raise ValueError('Nenhum documento encontrado.')
    for index, record in enumerate(records):
        for field in ('report', 'summary'):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f'Registro {index}: campo {field} vazio ou inválido.')
    return records


def load_govreport_records(limit=5, split='test', offset=0):
    if split not in ('train', 'validation', 'test') or limit < 1 or offset < 0:
        raise ValueError('Split, limite ou offset inválido.')
    from datasets import load_dataset
    stream = load_dataset(DATASET_NAME, 'document', split=split, streaming=True)
    records = list(islice(stream, offset, offset + limit))
    if len(records) != limit:
        raise ValueError(f'Esperados {limit} documentos no offset {offset}, recebidos {len(records)}.')
    return validate_records(records)


def load_local_records(path, limit=5, offset=0):
    if limit < 1 or offset < 0:
        raise ValueError('Limite ou offset inválido.')
    with Path(path).open(encoding='utf-8') as source:
        rows = (json.loads(line) for line in source if line.strip())
        return validate_records(list(islice(rows, offset, offset + limit)))
