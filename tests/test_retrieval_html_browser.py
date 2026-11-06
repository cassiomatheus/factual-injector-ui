"""Optional real Firefox tests with temporary datasets and databases."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from test_retrieval_html import ui, fixture


@unittest.skipUnless(os.environ.get('FACTUAL_INJECTOR_BROWSER_TESTS')=='1','Opt-in Firefox test')
class BrowserTests(unittest.TestCase):
    def setUp(self):
        from selenium import webdriver
        from selenium.webdriver.support.ui import WebDriverWait
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.input=self.root/'input.jsonl';fixture(self.input)
        self.original=self.input.read_bytes()
        self.server=ui.make_server(self.input,self.root/'reviews.sqlite3',0)
        self.worker=threading.Thread(target=self.server.serve_forever,daemon=True);self.worker.start()
        self.addCleanup(self.stop_server)
        options=webdriver.FirefoxOptions();options.add_argument('-headless')
        if os.environ.get('FACTUAL_INJECTOR_FIREFOX_BINARY'):
            options.binary_location=os.environ['FACTUAL_INJECTOR_FIREFOX_BINARY']
        self.browser=webdriver.Firefox(options=options);self.addCleanup(self.browser.quit)
        self.browser.set_window_size(1500,1000)
        self.browser.get(f'http://127.0.0.1:{self.server.server_port}/')
        self.wait=WebDriverWait(self.browser,15)

    def stop_server(self):
        self.server.shutdown();self.worker.join();self.server.server_close()

    def find(self,css): return self.browser.find_element('css selector',css)

    def click(self,css): self.browser.execute_script('arguments[0].click()',self.find(css))

    def fill(self,css,value):
        self.browser.execute_script("arguments[0].value=arguments[1];arguments[0].dispatchEvent(new Event('input',{bubbles:true}))",self.find(css),value)

    def saved(self): self.wait.until(lambda _:self.find('#saveStatus').text=='Salvo no computador ✓')

    def test_complete_resume_import_export_and_unchanged_source(self):
        self.assertEqual(len(self.browser.find_elements('css selector','input[type=radio]:checked')),0)
        self.assertFalse(self.find('#complete').is_enabled())
        self.click('input[name="evidence-0"][value="0"]')
        self.click('input[name="evidence-1"][value="2"]')
        self.fill('#reviewer','R1');self.fill('#notes','Evidência conferida')
        self.saved();self.click('#complete');self.saved()
        self.wait.until(lambda _:'doc1_sum_1' in self.find('#sentence').text)
        aligned=self.browser.execute_script('return aligned()')
        self.assertEqual(len(aligned),1)
        self.assertEqual(aligned[0]['validated_evidences'],[{'doc_sent_id':0,'text_en':'EPA reviewed reports.','human_label':'Apoio (Support)'}])
        self.assertEqual(len(aligned[0]['evidence_judgments']),2)
        exported=self.browser.execute_script('return payload()')
        backup=self.root/'backup.json';backup.write_text(json.dumps(exported))
        self.click('[data-index="0"]');self.fill('#notes','Nova observação');self.saved()
        self.assertEqual(self.server.store.snapshot()['doc1_sum_0']['status'],'draft')
        self.browser.execute_script('window.confirm=()=>true')
        self.find('#importFile').send_keys(str(backup));self.saved()
        self.wait.until(lambda _:self.server.store.snapshot()['doc1_sum_0']['status']=='completed')
        self.browser.refresh();self.click('[data-index="0"]')
        self.assertEqual(self.find('#notes').get_attribute('value'),'Evidência conferida')
        self.assertEqual(self.input.read_bytes(),self.original)

    def test_top_k_controls_preserve_hidden_labels_and_session_preference(self):
        rows, _ = ui.load_data(self.input)
        rows[0]['retrieved_top_k'] = [
            {'doc_sent_id': i, 'text_en': f'Evidence {i}', 'text_pt': f'Evidência {i}'}
            for i in range(6)]
        portable = self.root/'top-k.html'
        portable.write_text(ui.render_html(rows, 'top-k-fixture', {}))
        self.browser.get(portable.as_uri())
        count = lambda: len(self.browser.find_elements('css selector', '.evidence'))
        self.assertEqual(count(), 5)
        self.click('input[name="evidence-4"][value="0"]')
        self.click('#saveNow')
        before = self.browser.execute_script('return payload().reviews')
        self.click('[data-top-k="1"]')
        self.assertEqual(count(), 1)
        self.assertIn('ocultas', self.find('#topKStatus').text)
        self.assertFalse(self.find('#complete').is_enabled())
        self.click('[data-top-k="3"]')
        self.assertEqual(count(), 3)
        self.assertEqual(self.find('[data-top-k="3"]').get_attribute('aria-pressed'), 'true')
        self.click('[data-top-k="5"]')
        self.assertEqual(count(), 5)
        self.assertTrue(self.find('input[name="evidence-4"][value="0"]').is_selected())
        self.assertEqual(self.browser.execute_script('return payload().reviews'), before)
        self.click('[data-top-k="all"]')
        self.assertEqual(count(), 6)
        self.click('[data-top-k="3"]')
        self.browser.refresh()
        self.assertEqual(count(), 3)
        self.click('[data-index="1"]')
        self.assertEqual(count(), 2)
        self.assertIn('2 de 2', self.find('#topKStatus').text)
        self.assertTrue(self.find('[data-top-k="all"]').get_attribute('hidden'))

    def test_search_theme_navigation_portable_and_request_protection(self):
        from selenium.webdriver.support.ui import Select
        self.fill('#search','2 relatórios');self.assertIn('doc1_sum_1',self.find('#sentence').text)
        self.assertEqual(len(self.browser.find_elements('css selector','.queue-item')),1)
        self.fill('#search','');self.click('#theme')
        self.assertEqual(self.browser.execute_script('return document.documentElement.dataset.theme'),'dark')
        Select(self.find('#language')).select_by_value('pt')
        self.assertFalse(self.find('.claim [data-original]').is_displayed())
        self.click('.claim details summary')
        self.assertIn('EPA reviewed',self.find('.claim details').text)
        before=self.server.store.snapshot()
        self.browser.execute_script("document.activeElement.blur();document.dispatchEvent(new KeyboardEvent('keydown',{key:'j',bubbles:true}))")
        self.assertIn('doc1_sum_2',self.find('#sentence').text)
        self.assertEqual(self.server.store.snapshot(),before)
        code=self.browser.execute_async_script("const done=arguments[0];fetch('/api/reviews',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}).then(r=>done(r.status))")
        self.assertEqual(code,403)
        self.browser.set_window_size(390,900)
        self.assertFalse(self.browser.execute_script('return document.documentElement.scrollWidth>innerWidth'))
        rows,identity=ui.load_data(self.input);page=self.root/'portable.html';page.write_text(ui.render_html(rows,identity,{}))
        self.browser.get(page.as_uri());self.assertTrue(self.find('#offline').is_displayed())
        self.fill('#notes','Rascunho portátil');self.click('#saveNow');self.browser.refresh()
        self.assertEqual(self.find('#notes').get_attribute('value'),'Rascunho portátil')


if __name__=='__main__': unittest.main()
