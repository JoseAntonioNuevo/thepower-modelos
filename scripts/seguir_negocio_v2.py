"""Sigue un trabajo existente cada 45 s; conserva recibos, no crea recursos."""
import _bootstrap  # noqa: F401
import argparse
import json
import time
from datetime import datetime,timezone
from pathlib import Path
from curso.common import ROOT
from curso.negocio_v2 import frozen
from curso.remote import google_client


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',required=True)
    args=p.parse_args()
    frozen()
    if not args.job.startswith('projects/653878673654/locations/europe-west4/tuningJobs/'):
        p.error('El trabajo no pertenece al laboratorio autorizado')
    ev=ROOT/'material/evidencias/s8-gemini-negocio-v2-20261014'
    if (ev/'estado-final.json').exists():p.error('Ya existe un estado final; no se sobrescribe')
    snapshots=ev/'seguimiento';snapshots.mkdir(exist_ok=True)
    client=google_client('europe-west4')
    last=None;errors=0
    try:
        while True:
            try:job=client.tunings.get(name=args.job);errors=0
            except Exception as exc:
                errors+=1;print(f'Lectura fallida: {type(exc).__name__}, consecutivas={errors}',flush=True)
                if errors>=3:raise
                time.sleep(45);continue
            now=datetime.now(timezone.utc)
            record=json.loads(job.model_dump_json())
            path=snapshots/(now.strftime('%Y%m%dT%H%M%S')+'.json')
            with path.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
            signal=(record['state'],str(record.get('tuned_model')),str(record.get('error')))
            if signal!=last:
                print(json.dumps({'observed_at_utc':now.isoformat(),'state':record['state'],'tuned_model':record.get('tuned_model'),'error':record.get('error')},indent=2),flush=True);last=signal
            if record['state'] in ('JOB_STATE_SUCCEEDED','JOB_STATE_FAILED','JOB_STATE_CANCELLED'):
                with (ev/'estado-final.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
                if record['state']!='JOB_STATE_SUCCEEDED':raise RuntimeError(f"El trabajo terminó en {record['state']}")
                return
            time.sleep(45)
    finally:client.close()


if __name__=='__main__':main()
