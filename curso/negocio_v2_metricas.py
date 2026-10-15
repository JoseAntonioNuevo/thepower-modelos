"""Calibración v2.1 registrada sobre validación, antes de abrir test.

Conserva el evaluador congelado original. Acepta representaciones numéricas
equivalentes e inflexiones de aceptar; no cambia hechos, objetivos, estados,
referencias, umbrales ni las respuestas originales. Sigue siendo una medida
de cobertura, no un comprobador completo de contradicciones.
"""
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
from .common import ROOT
from .negocio_v2 import DATA,grade,mentions,normalized,sha

REGISTRY=ROOT/'material/evidencias/s8-gemini-negocio-v2-20261014/evaluacion-v21-registro.json'


def verify_registry():
    registry=json.loads(REGISTRY.read_text())
    if sha(Path(__file__))!=registry['metric_code_sha256']:raise ValueError('Calibración distinta de la registrada antes de test')
    if sha(DATA/'manifiesto.json')!=registry['dataset_manifest_sha256']:raise ValueError('Dataset distinto de la calibración')
    return registry


def equivalent(text,term):
    if mentions(text,term):return True
    value=normalized(term)
    if re.fullmatch(r'\d+(?:\.\d+)?',value):
        # 45,00 y 45 representan el mismo importe; 45,90 y 145 no.
        numbers=re.findall(r'(?<![\w.])\d+(?:[.,]\d+)?(?![\w.])',text)
        if any(Decimal(n.replace(',','.'))==Decimal(value) for n in numbers):return True
    if value in ('aceptacion','aceptada','aceptado'):
        return bool(re.search(r'\bacept(?:acion|ada|ado|a)\b',normalized(text)))
    return False


def grade_v21(raw,row,product_ids):
    result=grade(raw,row,product_ids)
    if not result['json_valido'] or not result['esquema_valido']:return result
    text=json.loads(raw)['respuesta']
    result['grupos_ausentes']=[group for group in row['criterios']['requiere'] if not any(equivalent(text,term) for term in group)]
    result['hechos_requeridos']=not result['grupos_ausentes']
    result['caso_completo']=all(result[k] for k in ('esquema_valido','estado_correcto','productos_correctos','hechos_requeridos'))
    return result


def calibrated_rows(rows):
    verify_registry()
    ids={p['id'] for p in json.loads((DATA/'catalogo.json').read_text())['productos']}
    return [{**r,'valoracion_legacy':r['valoracion'],'valoracion':grade_v21(r.get('respuesta_original'),r,ids)} for r in rows]
