import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('retrieval_html', ROOT/'app/retrieval_html.py')
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)


def fixture(path):
    rows = [{'doc_id': 'doc1', 'summary_sentence_id': f'doc1_sum_{i}',
             'summary_sentence_en': f'EPA reviewed {i+1} reports.',
             'summary_sentence_pt': f'EPA revisou {i+1} relatórios.',
             'retrieved_top_k': [{'doc_sent_id': 0, 'text_en': 'EPA reviewed reports.',
                                  'text_pt': 'EPA revisou relatórios.',
                                  'bi_encoder_score': .8, 'cross_encoder_score': 2.5},
                                 {'doc_sent_id': 1, 'text_en': 'Other evidence.', 'text_pt': 'Outra evidência.'}]}
            for i in range(3)]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return rows


def review(version=0, status='draft', labels=None):
    return {'version': version, 'status': status, 'labels': labels or {}, 'notes': 'Análise humana', 'reviewer': 'R1'}


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.input=self.root/'input.jsonl'
        self.rows=fixture(self.input); self.original=self.input.read_bytes()
        self.items,self.identity=ui.load_data(self.input)
        self.store=ui.ReviewStore(self.root/'reviews.sqlite3',self.items,self.identity)
        self.sid='doc1_sum_0'

    def test_draft_save_resume_and_source_integrity(self):
        saved=self.store.save_many({self.sid: review()})
        self.assertEqual(saved[self.sid]['labels'],{})
        resumed=ui.ReviewStore(self.store.path,self.items,self.identity)
        self.assertEqual(resumed.snapshot(),saved)
        self.assertEqual(self.input.read_bytes(),self.original)

    def test_completion_requires_explicit_choices(self):
        with self.assertRaises(ValueError):
            self.store.save_many({self.sid:review(status='completed')})
        saved=self.store.save_many({self.sid:review(status='completed',labels={'0':ui.LABELS[0],'1':ui.LABELS[2]})})
        self.assertEqual(saved[self.sid]['status'],'completed')

    def test_conflict_preserves_saved_review(self):
        self.store.save_many({self.sid:review()})
        with self.assertRaises(ui.ConflictError):
            self.store.save_many({self.sid:review()})
        self.assertEqual(self.store.snapshot()[self.sid]['version'],1)

    def test_bulk_import_rolls_back_on_conflict(self):
        self.store.save_many({'doc1_sum_1':review()})
        with self.assertRaises(ui.ConflictError):
            self.store.save_many({self.sid:review(),'doc1_sum_1':review()})
        self.assertNotIn(self.sid,self.store.snapshot())

    def test_invalid_label_and_foreign_evidence_rejected(self):
        for labels in [{'0':'SUPPORTED'},{'999':ui.LABELS[0]}]:
            with self.assertRaises(ValueError):
                self.store.save_many({self.sid:review(labels=labels)})
        self.assertEqual(self.store.snapshot(),{})

    def test_datasets_are_isolated(self):
        self.store.save_many({self.sid:review()})
        other=ui.ReviewStore(self.store.path,self.items,'other-dataset')
        self.assertEqual(other.snapshot(),{})

    def test_duplicate_summary_and_evidence_ids_rejected(self):
        for rows in [self.rows+[self.rows[0]], [{**self.rows[0],'retrieved_top_k':[self.rows[0]['retrieved_top_k'][0]]*2}]]:
            self.input.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError): ui.load_data(self.input)

    def test_html_preserves_data_and_escapes_script_closing(self):
        self.items[0]['summary_sentence_en']='</script><script>alert(1)</script>'
        page=ui.render_html(self.items,self.identity,{})
        self.assertNotIn(self.items[0]['summary_sentence_en'],page)
        self.assertIn('\\u003c/script',page)
        self.assertIn('id="evidences"',page)
        self.assertIn('id="offline"',page)
        self.assertNotIn('streamlit',page.lower())


if __name__=='__main__': unittest.main()
