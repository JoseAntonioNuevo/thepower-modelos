"""Pruebas de los límites del nuevo evaluador; no llaman a Google."""
import json
import unittest
from curso.negocio_v2 import grade,mentions


class BusinessContractTests(unittest.TestCase):
    def setUp(self):
        self.ids={'NH-L01','NH-L02'}
        self.row={'objetivo':{'estado':'respuesta','productos':['NH-L01']},'criterios':{'requiere':[['Órbita'],['39']],'max_palabras':140}}

    def answer(self,**kwargs):
        return json.dumps({'estado':'respuesta','productos':['NH-L01'],'respuesta':'Lámpara Órbita cuesta 39 €.',**kwargs},ensure_ascii=False)

    def test_required_price_cannot_match_another_number(self):
        for raw in ('139 euros','390 euros','39,90 euros'):
            self.assertFalse(mentions(raw,'39'),raw)
        self.assertTrue(mentions('El importe es 39.','39'))
        self.assertTrue(mentions('Cuesta 4,90 €','4.90'))

    def test_schema_correct_without_business_fact_is_not_success(self):
        score=grade(self.answer(respuesta='No conozco el precio de Órbita.'),self.row,self.ids)
        self.assertTrue(score['esquema_valido'])
        self.assertFalse(score['caso_completo'])

    def test_unknown_and_duplicate_product_ids_rejected(self):
        for ids in (['NH-X99'],['NH-L01','NH-L01']):
            self.assertFalse(grade(self.answer(productos=ids),self.row,self.ids)['esquema_valido'])

    def test_known_but_wrong_product_is_not_success(self):
        self.assertFalse(grade(self.answer(productos=['NH-L02']),self.row,self.ids)['caso_completo'])

    def test_markdown_is_not_repaired(self):
        self.assertFalse(grade('```json\n'+self.answer()+'\n```',self.row,self.ids)['json_valido'])

    def test_null_non_text_and_non_standard_json_fail(self):
        for raw in (None,'null',self.answer(respuesta=37),'{"estado": NaN}'):
            self.assertFalse(grade(raw,self.row,self.ids)['caso_completo'])


if __name__=='__main__':unittest.main()
