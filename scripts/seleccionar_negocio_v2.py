"""Candidatos espaciados y selección registrada antes de abrir test."""
import _bootstrap  # noqa: F401
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
from curso.common import ROOT,read_jsonl
from curso.negocio_v2 import DATA,frozen,sha
from curso.negocio_v2_metricas import calibrated_rows,verify_registry
from gemini_negocio_v2 import save_new,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('accion',choices=['candidatos','elegir'])
    p.add_argument('--estado',type=Path)
    args=p.parse_args()
    frozen()
    registry=verify_registry()
    ev=ROOT/'material/evidencias/s8-gemini-negocio-v2-20261014'
    if args.accion=='candidatos':
        if not args.estado:p.error('Indica el recibo real de trabajo terminado')
        job=json.loads(args.estado.read_text())
        if job['state']!='JOB_STATE_SUCCEEDED':p.error('El trabajo todavía no ha terminado correctamente')
        created=json.loads((ev/'trabajo-creado.json').read_text())
        if job['name']!=created['name']:p.error('El recibo no pertenece al nuevo trabajo v2')
        if job['supervised_tuning_spec']['training_dataset_uri']!='gs://thepower-s8-653878673654-eu/nebula-hogar-v2-20261014/train.jsonl':p.error('El trabajo no usa el dataset nuevo registrado')
        checkpoints=job['tuned_model']['checkpoints'] or [{'checkpoint_id':'final','epoch':8,'step':0,'endpoint':job['tuned_model']['endpoint']}]
        checkpoints=sorted(checkpoints,key=lambda x:x['step'])
        indexes=sorted({0,(len(checkpoints)-1)//2,len(checkpoints)-1})
        candidates=[checkpoints[i] for i in indexes]
        save_new(ev/'candidatos-checkpoint.json',{'job':job['name'],'registered_at_utc':datetime.now(timezone.utc).isoformat(),'rule':'first, middle, last by step; at most three','candidates':candidates,'all_checkpoint_count':len(checkpoints)})
        print(json.dumps(candidates,indent=2));return
    if list((ev/'test').glob('*.jsonl')):p.error('Test ya se ha consultado; no se cambia la selección')
    candidates=json.loads((ev/'candidatos-checkpoint.json').read_text())['candidates']
    expected=[r['id'] for r in read_jsonl(DATA/'validacion.jsonl')]
    results=[]
    for cp in candidates:
        path=ev/'validacion'/f"checkpoint-{cp['checkpoint_id']}.jsonl"
        rows=read_jsonl(path)
        if [r['id'] for r in rows]!=expected:raise ValueError('Validación incompleta o con IDs distintos')
        if any(r['manifiesto_sha256']!=sha(DATA/'manifiesto.json') or r['split']!='validacion' or r['model']!=cp['endpoint'] for r in rows):raise ValueError('Condiciones no coinciden con el experimento')
        results.append({**cp,'summary':summary(calibrated_rows(rows)),'legacy_summary':summary(rows),'validation_sha256':sha(path)})
    best=max(results,key=lambda x:(x['summary']['casos_completos'],x['step']))
    receipt={'selected_at_utc':datetime.now(timezone.utc).isoformat(),'endpoint':best['endpoint'],'checkpoint_id':best['checkpoint_id'],'epoch':best['epoch'],'step':best['step'],'validation':results,'rule':'max complete validation cases, calibrated v2.1; tie favors later step','test_opened':False,'manifest_sha256':sha(DATA/'manifiesto.json'),'assessment_id':registry['assessment_id']}
    save_new(ev/'seleccion-checkpoint.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
