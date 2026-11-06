"""Tradução em lotes, cache persistente e carregamento sob demanda."""
from contextlib import closing
from hashlib import sha256
from pathlib import Path
import sqlite3

TRANSLATION_MODEL = 'Helsinki-NLP/opus-mt-tc-big-en-pt'
LONG_TEXT = '[TRADUÇÃO NÃO GERADA: TEXTO MUITO LONGO; CONSULTE O ORIGINAL EN]'


class TranslationCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS translation_cache (cache_key TEXT PRIMARY KEY, translated_text TEXT NOT NULL)')

    @staticmethod
    def key(text, model):
        return sha256((model + '\0' + text).encode('utf-8')).hexdigest()

    def get(self, text, model):
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute('SELECT translated_text FROM translation_cache WHERE cache_key=?', (self.key(text, model),)).fetchone()
        return None if row is None else row[0]

    def save(self, mapping, model):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executemany('INSERT OR REPLACE INTO translation_cache VALUES (?, ?)',
                           [(self.key(text, model), translated) for text, translated in mapping.items()])


class CachedTranslator:
    def __init__(self, cache, device='cpu', model=TRANSLATION_MODEL, batch_size=4):
        self.cache, self.device, self.model = cache, device, model
        self.batch_size = batch_size
        self.pipeline = None

    def translate_many(self, texts):
        unique = list(dict.fromkeys(text for text in texts if text.strip()))
        result, missing = {}, []
        for text in unique:
            cached = self.cache.get(text, self.model)
            if cached is None:
                missing.append(text)
            else:
                result[text] = cached
        if not missing:
            return result
        if self.pipeline is None:
            from transformers import pipeline
            self.pipeline = pipeline('translation', model=self.model,
                                     device=0 if self.device == 'cuda' else -1)
        tokenizer = self.pipeline.tokenizer
        limit = min(getattr(tokenizer, 'model_max_length', 512), 512) - 64
        safe = []
        for text in missing:
            if len(tokenizer(text, truncation=False).input_ids) > limit:
                result[text] = LONG_TEXT
            else:
                safe.append(text)
        for start in range(0, len(safe), self.batch_size):
            batch = safe[start:start + self.batch_size]
            # Falhas interrompem a preparação; não armazenar inglês como tradução.
            outputs = self.pipeline(batch, batch_size=self.batch_size, max_length=400, truncation=False)
            if len(outputs) != len(batch):
                raise RuntimeError('Tradutor retornou quantidade inesperada de textos.')
            mapping = {text: output['translation_text'] for text, output in zip(batch, outputs)}
            self.cache.save(mapping, self.model)
            result.update(mapping)
        return result
