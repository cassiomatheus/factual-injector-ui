#!/usr/bin/env python3
"""Auditoria HTML de evidências, sem Streamlit ou carregamento de modelos."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import sqlite3
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('Apoio (Support)', 'Contradição (Contradiction)', 'Nenhum/Irrelevante')


def load_data(path):
    raw = Path(path).read_bytes()
    items = []
    seen = set()
    for number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        for field in ('doc_id', 'summary_sentence_id', 'summary_sentence_en'):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f'Linha {number}: {field} inválido')
        if item['summary_sentence_id'] in seen:
            raise ValueError('summary_sentence_id duplicado')
        seen.add(item['summary_sentence_id'])
        if not isinstance(item.get('retrieved_top_k'), list):
            raise ValueError('retrieved_top_k deve ser uma lista')
        ids = set()
        for evidence in item['retrieved_top_k']:
            sid = evidence.get('doc_sent_id')
            if isinstance(sid, bool) or not isinstance(sid, (str, int)):
                raise ValueError('doc_sent_id inválido')
            if str(sid) in ids or not isinstance(evidence.get('text_en'), str) or not evidence['text_en'].strip():
                raise ValueError('Evidência vazia ou identificador duplicado na sentença')
            ids.add(str(sid))
        items.append(item)
    if not items:
        raise ValueError('Arquivo sem sentenças para revisão')
    return items, sha256(raw).hexdigest()


class ConflictError(ValueError):
    pass


class ReviewStore:
    def __init__(self, path, items, dataset_id):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items = {i['summary_sentence_id']: i for i in items}
        self.dataset_id = dataset_id
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS reviews (dataset TEXT, sid TEXT, version INTEGER, body TEXT, PRIMARY KEY(dataset,sid))')
            db.execute('CREATE TABLE IF NOT EXISTS review_events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, dataset TEXT NOT NULL, sid TEXT NOT NULL, version INTEGER NOT NULL, body TEXT NOT NULL, UNIQUE(dataset,sid,version))')
            # Preserva como ponto inicial o estado existente antes da migração.
            db.execute('INSERT OR IGNORE INTO review_events(dataset,sid,version,body) SELECT dataset,sid,version,body FROM reviews')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            db.execute('PRAGMA synchronous=FULL')
            with db:
                yield db
        finally:
            db.close()

    def snapshot(self):
        with self.connect() as db:
            return {sid: dict(json.loads(body), version=version) for sid, version, body in db.execute(
                'SELECT sid,version,body FROM reviews WHERE dataset=?', (self.dataset_id,))}

    def history(self):
        """Histórico da entrada atual; revisões anteriores à migração têm um snapshot."""
        with self.connect() as db:
            rows = db.execute('SELECT event_id,sid,version,body FROM review_events WHERE dataset=? ORDER BY event_id', (self.dataset_id,)).fetchall()
        return [{**json.loads(body), 'event_id': eid, 'summary_sentence_id': sid,
                 'version': version} for eid, sid, version, body in rows]

    def validate(self, sid, review):
        if sid not in self.items or not isinstance(review, dict):
            raise ValueError('Sentença desconhecida ou revisão inválida')
        item = self.items[sid]
        labels = review.get('labels', {})
        ids = {str(e['doc_sent_id']) for e in item['retrieved_top_k']}
        if not isinstance(labels, dict) or set(labels) - ids or any(v not in LABELS for v in labels.values()):
            raise ValueError('Identificador de evidência ou rótulo inválido')
        status = review.get('status', 'draft')
        if status not in ('draft', 'completed'):
            raise ValueError('Estado de revisão inválido')
        if status == 'completed' and set(labels) != ids:
            raise ValueError('Avalie explicitamente todas as evidências antes de concluir')
        for field in ('notes', 'reviewer'):
            if not isinstance(review.get(field, ''), str) or len(review.get(field, '')) > 20000:
                raise ValueError(f'{field} inválido')
        return {'labels': labels, 'status': status, 'notes': review.get('notes', ''),
                'reviewer': review.get('reviewer', ''),
                'updated_utc': datetime.now(timezone.utc).isoformat()}

    def save_many(self, updates):
        clean = []
        for sid, record in updates.items():
            version = record.get('version', 0)
            if isinstance(version, bool) or not isinstance(version, int) or version < 0:
                raise ValueError('Versão inválida')
            clean.append((sid, self.validate(sid, record), version))
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for sid, review, expected in clean:
                row = db.execute('SELECT version FROM reviews WHERE dataset=? AND sid=?', (self.dataset_id, sid)).fetchone()
                if (row[0] if row else 0) != expected:
                    raise ConflictError('Outra aba alterou esta análise. Exporte uma cópia antes de recarregar.')
                version = expected + 1
                db.execute('INSERT INTO review_events(dataset,sid,version,body) VALUES (?,?,?,?)',
                           (self.dataset_id, sid, version, json.dumps(review, ensure_ascii=False, allow_nan=False)))
                db.execute('INSERT OR REPLACE INTO reviews VALUES (?,?,?,?)',
                           (self.dataset_id, sid, version, json.dumps(review, ensure_ascii=False, allow_nan=False)))
        return self.snapshot()


def render_html(items, dataset_id, reviews, token=None):
    payload = {'schema_version': 1, 'dataset_id': dataset_id, 'items': items, 'reviews': reviews, 'token': token}
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return (Path(__file__).parent/'templates/retrieval.html').read_text(encoding='utf-8').replace('__BOOTSTRAP_JSON__', encoded)


def make_server(input_path, database, port=8766):
    items, identity = load_data(input_path)
    store = ReviewStore(database, items, identity)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def respond(self, code, payload, kind='application/json; charset=utf-8'):
            body = payload.encode('utf-8') if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def local_request(self):
            host = f'127.0.0.1:{self.server.server_port}'
            return self.headers.get('Host') == host and self.headers.get('Origin', f'http://{host}') == f'http://{host}'

        def do_GET(self):
            if not self.local_request():
                self.respond(403, {'error': 'Acesso permitido somente pela origem local'})
            elif self.path == '/':
                self.respond(200, render_html(items, identity, store.snapshot(), token), 'text/html; charset=utf-8')
            else:
                self.respond(404, {'error': 'Página não encontrada'})

        def do_POST(self):
            if not self.local_request() or self.headers.get('X-Review-Token') != token:
                self.respond(403, {'error': 'Recarregue a página pelo endereço local do serviço'})
                return
            if self.path != '/api/reviews':
                self.respond(404, {'error': 'Rota não encontrada'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4*1024*1024 or self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('Requisição inválida ou maior que 4 MiB')
                payload = json.loads(self.rfile.read(length))
                if payload.get('dataset_id') != identity or not isinstance(payload.get('reviews'), dict):
                    raise ValueError('Revisão de outro arquivo de entrada')
                result = store.save_many(payload['reviews'])
                self.respond(200, {'reviews': result})
            except ConflictError as exc:
                self.respond(409, {'error': str(exc)})
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                self.respond(400, {'error': str(exc)})
            except (OSError, sqlite3.Error):
                self.respond(500, {'error': 'Falha ao gravar no computador. Exporte uma cópia e tente novamente.'})

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.store = store
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/processed/retrieval_ready.jsonl')
    parser.add_argument('--database', type=Path, default=ROOT/'data/annotations/retrieval.sqlite3')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--open', action='store_true')
    parser.add_argument('--export-history', type=Path, help='Exporta histórico da entrada em JSON e encerra')
    parser.add_argument('--export-html', type=Path, help='Gera HTML portátil; nesse modo as notas ficam no navegador até exportar JSON')
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('Porta inválida')
    try:
        if args.export_history:
            items, identity = load_data(args.input)
            store = ReviewStore(args.database, items, identity)
            args.export_history.parent.mkdir(parents=True, exist_ok=True)
            with args.export_history.open('x', encoding='utf-8') as target:
                json.dump({'schema_version': 1, 'dataset_id': identity, 'events': store.history()}, target, ensure_ascii=False, indent=2)
            print('Histórico exportado:', args.export_history.resolve())
            return 0
        if args.export_html:
            items, identity = load_data(args.input)
            store = ReviewStore(args.database, items, identity)
            if args.export_html.exists():
                parser.error('Destino já existe. Escolha outro nome para preservar o HTML anterior.')
            args.export_html.parent.mkdir(parents=True, exist_ok=True)
            args.export_html.write_text(render_html(items, identity, store.snapshot()), encoding='utf-8')
            print('HTML portátil:', args.export_html.resolve())
            return 0
        server = make_server(args.input, args.database, args.port)
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.exit(1, f'Não foi possível iniciar: {exc}\nPrepare a entrada com python src/pre_processor.py --limit 5 --device auto\n')
    url = f'http://127.0.0.1:{server.server_port}/'
    print('Mapeamento de evidências:', url, flush=True)
    print('Análises no computador:', server.store.path, flush=True)
    print('Mantenha este terminal aberto. Ctrl+C encerra; as análises salvas permanecem.', flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
