import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from curso.common import ROOT,assert_frozen,read_jsonl,validate_answer,validate_splits,messages,write_jsonl
from curso.remote import gemini_record,google_client
from gemini_tuning import job_spec
from resumir import summarize
from unittest.mock import patch
from entrenar_lora import training_output


class Contracts(unittest.TestCase):
    def setUp(self):self.expected={'categoria':'acceso','prioridad':'media','respuesta':'No compartas contraseñas.'}
    def test_json_strict_no_fences(self):
        self.assertFalse(validate_answer('```json\n'+json.dumps(self.expected)+'\n```',self.expected)['json_valido'])
        self.assertFalse(validate_answer('{"categoria":NaN}',self.expected)['json_valido'])
    def test_exact_fields_and_enum(self):
        self.assertTrue(validate_answer(json.dumps(self.expected),self.expected)['esquema_valido'])
        wrong={**self.expected,'otra':1}
        self.assertFalse(validate_answer(json.dumps(wrong),self.expected)['esquema_valido'])
    def test_unparseable_is_in_denominator(self):
        rows=[{'valoracion':validate_answer(json.dumps(self.expected),self.expected),'latencia_s':1,'coste_usd':None},
              {'valoracion':validate_answer('error',self.expected),'latencia_s':None,'coste_usd':None,'error':'TimeoutError'}]
        out=summarize(rows);self.assertEqual((out['casos'],out['categoria_correcta']),(2,1));self.assertIsNone(out['coste_usd'])
    def test_splits_and_freeze(self):
        self.assertEqual(validate_splits(),{'train':240,'validacion':30,'test':30});self.assertTrue(assert_frozen())
    def test_duplicate_family_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for s in ('train','validacion','test'):
                rows=read_jsonl(ROOT/f'datos/{s}.jsonl')
                if s=='test':rows[0]['familia']='train-acceso-00'
                write_jsonl(Path(tmp)/f'{s}.jsonl',rows)
            with self.assertRaisesRegex(ValueError,'Familia'):validate_splits(tmp)
    def test_fewshot_only_from_training(self):
        msg=messages('CONSULTA NUEVA',True)
        self.assertEqual(msg[-1]['content'],'CONSULTA NUEVA');self.assertEqual(len(msg),10)
        training={r['incidencia'] for r in read_jsonl(ROOT/'datos/train.jsonl')}
        self.assertTrue(all(m['content'] in training for m in msg[1:-1] if m['role']=='user'))
    def test_gemini_conversion_preserves_labels(self):
        row=read_jsonl(ROOT/'datos/test.jsonl')[0];record=gemini_record(row)
        self.assertEqual(record['contents'][0]['parts'][0]['text'],row['incidencia'])
        self.assertEqual(json.loads(record['contents'][1]['parts'][0]['text']),row['objetivo'])
        self.assertNotIn('id',record)
    def test_gemini_id_mapping(self):
        ids=json.loads((ROOT/'datos/gemini/ids.json').read_text())
        for s in ('train','validacion','test'):
            self.assertEqual(ids[s]['ids_por_linea'],[r['id'] for r in read_jsonl(ROOT/f'datos/{s}.jsonl')])
    def test_missing_project_prevents_client(self):
        with patch.dict('os.environ',{},clear=True),patch('curso.remote.load_config'):
            with self.assertRaisesRegex(ValueError,'laboratorio'):google_client()
    def test_job_spec_never_includes_test(self):
        spec=job_spec('curso-demo','bucket-demo')
        self.assertNotIn('test',json.dumps(spec));self.assertEqual(spec['location'],'europe-west4')
        with self.assertRaises(ValueError):job_spec(None,'bucket')
    def test_imports_do_not_connect(self):
        import importlib
        with patch('requests.post',side_effect=AssertionError('network')):
            importlib.reload(sys.modules['curso.remote'])

    def test_training_does_not_overwrite_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'adapter';path.mkdir()
            self.assertEqual(training_output(path),path.resolve())
            artifact=path/'adapter_model.safetensors';artifact.write_bytes(b'preserve')
            with self.assertRaisesRegex(ValueError,'Salida no vacía'):training_output(path)
            self.assertEqual(artifact.read_bytes(),b'preserve')


if __name__=='__main__':unittest.main()
