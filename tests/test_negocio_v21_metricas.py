"""Regresiones de falsos negativos encontrados en validación, no en test."""
import unittest
from curso.negocio_v2_metricas import equivalent


class MetricCalibrationTests(unittest.TestCase):
    def test_money_same_value_different_scale(self):
        self.assertTrue(equivalent('Subtotal: 45,00 €; total: 49,90 €.','45'))
        self.assertTrue(equivalent('Subtotal: 25.00 euros.','25'))
        for text in ('145 euros','45,90 euros','45,001 euros'):
            self.assertFalse(equivalent(text,'45'),text)

    def test_acceptance_inflection(self):
        self.assertTrue(equivalent('desde que se acepta la devolución','aceptación'))
        self.assertFalse(equivalent('desde que se envía la devolución','aceptación'))

    def test_calibration_does_not_replace_missing_business_fact(self):
        self.assertFalse(equivalent('El envío no tiene información disponible','4,90'))


if __name__=='__main__':unittest.main()
