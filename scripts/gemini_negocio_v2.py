"""Preparar, subir, entrenar y evaluar el experimento independiente de negocio.

No modifica datos, prompts, trabajo ni resultados del clasificador de S7/S8.
Cada acción es explícita. Los archivos de recibos y resultados nunca se pisan.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import math
import os
import statistics
from datetime import datetime,timezone
from pathlib import Path

from curso.common import ROOT,read_jsonl
from curso.negocio_v2 import DATA,Predictor,frozen,grade,sha
from curso.remote import google_client,load_config

EVIDENCE=ROOT/'material/evidencias/s8-gemini-negocio-v2-20261014'


def save_new(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')


def summary(rows):
    def tokens(key):
        values=[r.get(key) for r in rows]
        return sum(values) if values and all(isinstance(v,int) for v in values) else None
    return {'casos':len(rows),'casos_completos':sum(r['valoracion']['caso_completo'] for r in rows),
        **{k:sum(r['valoracion'][k] for r in rows) for k in ('json_valido','esquema_valido','estado_correcto','productos_correctos','hechos_requeridos')},
        'errores_api':sum(bool(r.get('error')) for r in rows),
        'input_tokens':tokens('tokens_entrada'),
        'output_tokens':tokens('tokens_salida'),
        'latencia_mediana_s':statistics.median(r['latencia_s'] for r in rows if r.get('latencia_s') is not None) if any(r.get('latencia_s') is not None for r in rows) else None,
        'ids_incompletos':[r['id'] for r in rows if not r['valoracion']['caso_completo']]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('accion',choices=['comprobar','contar-tokens','subir-datos','crear','estado','checkpoints','evaluar','resumir'])
    p.add_argument('--datos',type=Path,default=DATA)
    p.add_argument('--evidencias',type=Path,default=EVIDENCE)
    p.add_argument('--job')
    p.add_argument('--modelo')
    p.add_argument('--split',choices=['desarrollo','validacion','test'],default='validacion')
    p.add_argument('--condicion',choices=['base_sin_catalogo','ajustado_sin_catalogo','base_con_catalogo_control'])
    p.add_argument('--salida',type=Path)
    p.add_argument('--presupuesto-usd',type=float)
    args=p.parse_args()
    manifest=frozen(args.datos)
    protocol=json.loads((args.datos/'protocolo.json').read_text())
    catalog=json.loads((args.datos/'catalogo.json').read_text())
    ids={x['id'] for x in catalog['productos']}
    ev=args.evidencias;ev.mkdir(parents=True,exist_ok=True)
    if args.accion=='comprobar':
        rows=[r for split in manifest['splits'] for r in read_jsonl(args.datos/f'{split}.jsonl')]
        bad=[r['id'] for r in rows if not grade(json.dumps(r['objetivo'],ensure_ascii=False),r,ids)['caso_completo']]
        if bad:raise ValueError(f'Objetivos que no pasan su contrato: {bad}')
        print(json.dumps({'estado':'OFFLINE OK','manifest':manifest,'objetivos_comprobados':len(rows),'catalogo_productos':len(ids)},ensure_ascii=False,indent=2));return
    if args.accion=='resumir':
        if not args.salida:p.error('Indica --salida del JSONL a resumir')
        print(json.dumps(summary(read_jsonl(args.salida)),ensure_ascii=False,indent=2));return
    load_config()
    if os.getenv('GOOGLE_CLOUD_PROJECT')!=protocol['project']:raise ValueError('Proyecto local distinto del protocolo')
    if args.accion in ('estado','checkpoints'):
        if not args.job:p.error('Indica --job real existente')
        client=google_client(protocol['location'])
        try:job=client.tunings.get(name=args.job)
        finally:client.close()
        receipt=json.loads(job.model_dump_json())
        if args.accion=='estado':
            if args.salida:save_new(args.salida,receipt)
            print(json.dumps(receipt,ensure_ascii=False,indent=2))
        else:
            print(json.dumps(receipt.get('tuned_model',{}),ensure_ascii=False,indent=2))
        return
    if args.accion=='evaluar':
        if not args.condicion or not args.salida:p.error('Indica --condicion y --salida nueva')
        if args.salida.exists():p.error('Salida existente; conserva los resultados')
        if args.condicion=='ajustado_sin_catalogo' and not args.modelo:p.error('Indica endpoint real ajustado')
        if args.split=='test' and not (ev/'seleccion-checkpoint.json').exists():p.error('Selecciona y registra el checkpoint antes de abrir test')
        model=args.modelo or protocol['base_model']
        if args.condicion!='ajustado_sin_catalogo' and model!=protocol['base_model']:p.error('La base del experimento está fijada en el protocolo')
        if args.split=='test' and args.condicion=='ajustado_sin_catalogo':
            selected=json.loads((ev/'seleccion-checkpoint.json').read_text())
            if model!=selected['endpoint']:p.error('Test usa el checkpoint registrado antes de abrirlo')
        with_catalog=args.condicion=='base_con_catalogo_control'
        rows=read_jsonl(args.datos/f'{args.split}.jsonl')
        args.salida.parent.mkdir(parents=True,exist_ok=True)
        predictor=Predictor()
        try:
            with args.salida.open('x') as f:
                for row in rows:
                    try:result=predictor.predict(row['pregunta'],model,with_catalog,args.datos);error=None
                    except Exception as exc:result={'respuesta_original':None,'latencia_s':None,'input_tokens':None,'output_tokens':None,'model':model};error=f'{type(exc).__name__}: {exc}'
                    item={**row,**result,'condicion':args.condicion,'split':args.split,'error':error,'fecha_utc':datetime.now(timezone.utc).isoformat(),'manifiesto_sha256':sha(args.datos/'manifiesto.json'),'valoracion':grade(result.get('respuesta_original'),row,ids)}
                    f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush()
                    print(row['id'],args.condicion,'OK' if item['valoracion']['caso_completo'] else 'INCOMPLETO',f'error={error}',flush=True)
        finally:predictor.close()
        print(json.dumps(summary(read_jsonl(args.salida)),ensure_ascii=False,indent=2));return
    if args.accion=='contar-tokens':
        # Una lectura del contador; agregamos contenido y margen de encuadre por ejemplo.
        records=read_jsonl(args.datos/'gemini/train.jsonl')
        text='\n'.join(r['systemInstruction']['parts'][0]['text']+'\n'+r['contents'][0]['parts'][0]['text']+'\n'+r['contents'][1]['parts'][0]['text'] for r in records)
        client=Predictor()
        try:count=client.count_tokens(text)
        finally:client.close()
        upper=math.ceil(count*1.1)+len(records)*30
        estimate={'count_tokens_aggregated':count,'estimated_training_token_upper':upper,'epoch_count':protocol['epochs'],'training_usd_upper_at_3_per_mtok':upper*protocol['epochs']*3/1e6,'additional_inference_reserve_usd':0.5,'previous_estimated_spend_usd':protocol['previous_estimated_spend_usd'],'billing_note':'Conservador; la cifra final del job y la factura se registran aparte.'}
        estimate['combined_estimated_upper_usd']=estimate['training_usd_upper_at_3_per_mtok']+0.5+protocol['previous_estimated_spend_usd']
        save_new(ev/'presupuesto-previo.json',estimate);print(json.dumps(estimate,indent=2));return
    if args.presupuesto_usd!=protocol['authorized_budget_usd']:p.error('Usa el presupuesto explícito del protocolo: 5 USD acumulados')
    budget=json.loads((ev/'presupuesto-previo.json').read_text())
    if budget['combined_estimated_upper_usd']>args.presupuesto_usd:raise ValueError('Estimación fuera del presupuesto; no se sube ni crea un job')
    if args.accion=='subir-datos':
        import google.auth
        from google.cloud import storage
        creds,_=google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        if hasattr(creds,'with_quota_project'):creds=creds.with_quota_project(protocol['project'])
        client=storage.Client(project=protocol['project'],credentials=creds)
        try:
            bucket=client.get_bucket(protocol['bucket'])
            if not bucket.iam_configuration.uniform_bucket_level_access_enabled or bucket.iam_configuration.public_access_prevention!='enforced':raise ValueError('Bucket no conserva los controles privados previstos')
            uploaded=[]
            for split in ('train','validacion'):
                path=args.datos/'gemini'/f'{split}.jsonl';name=f"{protocol['gcs_prefix']}/{split}.jsonl";blob=bucket.blob(name)
                blob.upload_from_filename(path,content_type='application/jsonl',if_generation_match=0)
                if __import__('hashlib').sha256(blob.download_as_bytes()).hexdigest()!=sha(path):raise ValueError('SHA256 de subida no coincide')
                uploaded.append({'object':name,'generation':blob.generation,'sha256':sha(path),'size':blob.size})
            objects=[b.name for b in client.list_blobs(bucket,prefix=protocol['gcs_prefix']+'/')]
            if len(objects)!=2:raise ValueError('El prefijo nuevo debe contener solo train y validación')
            receipt={'bucket':bucket.name,'location':bucket.location,'public_access_prevention':'enforced','uniform_access':True,'objects':uploaded,'test_uploaded':False}
            save_new(ev/'datos-subidos.json',receipt);print(json.dumps(receipt,indent=2))
        finally:client.close()
        return
    if args.accion=='crear':
        from google.genai import types
        uploaded=json.loads((ev/'datos-subidos.json').read_text())
        for row in uploaded['objects']:
            split=Path(row['object']).stem
            if row['sha256']!=sha(args.datos/'gemini'/f'{split}.jsonl'):raise ValueError('Subida y manifiesto no coinciden')
        config=types.CreateTuningJobConfig(tuned_model_display_name='thepower-nebula-hogar-v2',epoch_count=protocol['epochs'],export_last_checkpoint_only=False,validation_dataset=types.TuningDataset(gcs_uri=f"gs://{protocol['bucket']}/{protocol['gcs_prefix']}/validacion.jsonl"))
        # El recibo previo evita duplicar un job tras un resultado de red desconocido.
        save_new(ev/'creacion-solicitada.json',{'config':config.model_dump(mode='json',exclude_none=True),'project':protocol['project'],'time_utc':datetime.now(timezone.utc).isoformat(),'budget':budget})
        client=google_client(protocol['location'])
        try:job=client.tunings.tune(base_model=protocol['base_model'],training_dataset=types.TuningDataset(gcs_uri=f"gs://{protocol['bucket']}/{protocol['gcs_prefix']}/train.jsonl"),config=config)
        finally:client.close()
        save_new(ev/'trabajo-creado.json',json.loads(job.model_dump_json()))
        print(json.dumps({'job':job.name,'state':str(job.state),'budget_usd':args.presupuesto_usd}));return


if __name__=='__main__':main()
