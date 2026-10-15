"""Test nuevo para la consulta de precios, elegido por diagnóstico de validación.

Usa el mismo entrenamiento real y el mismo prompt. Conserva el test amplio y
no lo transforma ni selecciona sus mejores respuestas. Cada condición tiene
una pasada nueva sobre 56 preguntas registradas, de ocho precios independientes.
"""
import _bootstrap  # noqa: F401
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
from curso.common import ROOT,read_jsonl
from curso.negocio_v2 import DATA,Predictor,frozen,sha
from curso.negocio_v2_metricas import grade_v21,verify_registry
from gemini_negocio_v2 import summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condicion',required=True,choices=['base_sin_catalogo','ajustado_sin_catalogo','base_con_catalogo_control'])
    p.add_argument('--por-nombre',action='store_true',help='Evaluación nueva de la atención comercial por nombre del producto')
    args=p.parse_args()
    frozen();verify_registry()
    prices=ROOT/('experimentos/nebula-hogar-precios-nombre-v2' if args.por_nombre else 'experimentos/nebula-hogar-precios-v2')
    manifest=json.loads((prices/'manifiesto.json').read_text())
    for name,expected in manifest.items():
        if sha(prices/name)!=expected:raise ValueError('Test de precios modificado desde su registro')
    ev=ROOT/'material/evidencias/s8-gemini-negocio-v2-20261014'
    selected=json.loads((ev/'seleccion-checkpoint.json').read_text())
    model=selected['endpoint'] if args.condicion=='ajustado_sin_catalogo' else 'gemini-3.1-flash-lite'
    target=ev/('precios-por-nombre' if args.por_nombre else 'precios')/f'{args.condicion}.jsonl'
    if target.exists():p.error('Salida existente; no se repite ni sobrescribe el test')
    target.parent.mkdir(exist_ok=True)
    ids={p['id'] for p in json.loads((DATA/'catalogo.json').read_text())['productos']}
    client=Predictor()
    try:
        with target.open('x') as f:
            for row in read_jsonl(prices/'test.jsonl'):
                try:response=client.predict(row['pregunta'],model,args.condicion=='base_con_catalogo_control');error=None
                except Exception as exc:response={'respuesta_original':None,'latencia_s':None,'model':model};error=f'{type(exc).__name__}: {exc}'
                result={**row,**response,'error':error,'split':'test_precios','condicion':args.condicion,'fecha_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':sha(prices/'manifiesto.json'),'valoracion':grade_v21(response.get('respuesta_original'),row,ids)}
                f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
                print(row['id'],args.condicion,result['valoracion']['caso_completo'],flush=True)
    finally:client.close()
    print(json.dumps(summary(read_jsonl(target)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
