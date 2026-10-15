"""Consulta real del negocio para la clase; usa la selección o el modelo base.

Consume inferencia. Conserva una salida nueva si se indica --salida, sin
modificar datos, evaluación ni resultados previos.
"""
import _bootstrap  # noqa: F401
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
from curso.common import ROOT
from curso.negocio_v2 import Predictor,frozen,sha,DATA


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pregunta',required=True)
    p.add_argument('--base',action='store_true')
    p.add_argument('--con-catalogo',action='store_true')
    p.add_argument('--salida',type=Path)
    args=p.parse_args()
    frozen()
    if args.con_catalogo and not args.base:p.error('El control con catálogo se demuestra con --base')
    if args.salida and args.salida.exists():p.error('Salida existente; conserva el recibo')
    ev=ROOT/'material/evidencias/s8-gemini-negocio-v2-20261014'
    model='gemini-3.1-flash-lite' if args.base else json.loads((ev/'seleccion-checkpoint.json').read_text())['endpoint']
    client=Predictor()
    try:result=client.predict(args.pregunta,model,args.con_catalogo)
    finally:client.close()
    receipt={'pregunta':args.pregunta,'fecha_utc':datetime.now(timezone.utc).isoformat(),'manifiesto_sha256':sha(DATA/'manifiesto.json'),**result}
    if args.salida:
        args.salida.parent.mkdir(parents=True,exist_ok=True)
        with args.salida.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
