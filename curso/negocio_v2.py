"""Contrato independiente del experimento de preguntas sobre Nébula Hogar.

Conserva el texto original. El comprobador mide cobertura de hechos requeridos;
la revisión contra catálogo sigue siendo necesaria para detectar contradicciones.
No cambia el clasificador de incidencias original del curso.
"""
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

from .common import ROOT, assert_frozen, read_jsonl
from .http import bounded_json
from .remote import gemini_reply, load_config, vertex_host

DATA=ROOT/'experimentos/nebula-hogar-v2'
STATES={'respuesta','aclaracion','no_catalogo','sin_datos','fuera_ambito'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen(data=DATA):
    """Comprueba ambos experimentos; nunca arregla un manifiesto que falla."""
    assert_frozen()
    manifest=json.loads((data/'manifiesto.json').read_text())
    for relative,expected in manifest['files_sha256'].items():
        if sha(data/relative)!=expected:raise ValueError(f'Datos v2 modificados: {relative}')
    if manifest.get('codigo_evaluador_sha256')!=sha(ROOT/'curso/negocio_v2.py'):
        raise ValueError('El evaluador difiere del contrato congelado v2')
    return manifest


def normalized(text):
    text=unicodedata.normalize('NFKD',text.casefold())
    text=''.join(c for c in text if not unicodedata.combining(c))
    text=re.sub(r'(?<=\d),(?=\d)', '.', text)
    text=text.replace('vatios','w').replace('mililitros','ml')
    return re.sub(r'\s+',' ',re.sub(r'[^\w.]+',' ',text)).strip()


def mentions(text,term):
    if term=='?':return '?' in text
    value=normalized(term)
    variants={value}
    for word,digit in [('tres','3'),('cinco','5'),('dos','2'),('seis','6'),('doce','12')]:
        if word in value:variants.add(value.replace(word,digit))
    for variant in variants:
        # Un punto final no forma parte de un nombre. En números sí hay que
        # impedir que 9 coincida con 19, 90 o 9,90.
        pattern=(r'(?<![\d.])'+re.escape(variant)+r'(?!\d|\.\d)'
                 if re.fullmatch(r'\d+(?:\.\d+)?',variant)
                 else r'(?<!\w)'+re.escape(variant)+r'(?!\w)')
        if re.search(pattern,normalized(text)):
            return True
    return False


def grade(raw,row,product_ids):
    """JSON estricto, sin extracción de Markdown ni reparación de la respuesta."""
    result={'json_valido':False,'esquema_valido':False,'estado_correcto':False,'productos_correctos':False,'hechos_requeridos':False,'caso_completo':False,'grupos_ausentes':[],'motivo':None}
    if not isinstance(raw,str) or len(raw)>16384:
        result['motivo']='Sin texto o demasiado largo';return result
    try:
        obj=json.loads(raw,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    except (ValueError,TypeError,RecursionError):
        result['motivo']='JSON inválido';return result
    result['json_valido']=True
    if not isinstance(obj,dict) or set(obj)!={'estado','productos','respuesta'}:
        result['motivo']='Campos incorrectos';return result
    ids=obj['productos'];text=obj['respuesta'];state=obj['estado']
    schema=(isinstance(state,str) and state in STATES and isinstance(ids,list)
            and all(isinstance(x,str) and x in product_ids for x in ids)
            and len(ids)==len(set(ids)) and isinstance(text,str)
            and 0<len(text.split())<=row['criterios']['max_palabras'])
    result['esquema_valido']=schema
    result['estado_correcto']=state==row['objetivo']['estado']
    result['productos_correctos']=isinstance(ids,list) and all(isinstance(x,str) for x in ids) and sorted(ids)==sorted(row['objetivo']['productos'])
    if isinstance(text,str):
        result['grupos_ausentes']=[group for group in row['criterios']['requiere'] if not any(mentions(text,term) for term in group)]
        result['hechos_requeridos']=not result['grupos_ausentes']
    result['caso_completo']=all(result[k] for k in ('esquema_valido','estado_correcto','productos_correctos','hechos_requeridos'))
    return result


def payload(question,with_catalog=False,data=DATA):
    system=(data/'prompt.txt').read_text().strip()
    if with_catalog:
        system+='\nCONTEXTO AUTORIZADO DEL NEGOCIO:\n'+(data/'catalogo.json').read_text()
    generation=json.loads((data/'protocolo.json').read_text())['generation']
    return {'systemInstruction':{'parts':[{'text':system}]},
            'contents':[{'role':'user','parts':[{'text':question}]}],
            'generationConfig':generation}


class Predictor:
    """ADC existente, cuota del proyecto autorizado y cero reintentos correctores."""
    def __init__(self):
        import google.auth
        import os
        from google.auth.transport.requests import AuthorizedSession
        load_config()
        project=os.getenv('GOOGLE_CLOUD_PROJECT')
        if project!='gen-lang-client-0913716113':raise ValueError('Proyecto distinto del laboratorio autorizado')
        creds,_=google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        if hasattr(creds,'with_quota_project'):creds=creds.with_quota_project(project)
        self.project=project
        self.session=AuthorizedSession(creds,max_refresh_attempts=0)

    def close(self):self.session.close()

    def count_tokens(self,text):
        # El SDK 1.42 construye un host incorrecto para la multirregión eu.
        # Usamos el mismo host REST correcto que generateContent, sin generar.
        resource=f'projects/{self.project}/locations/eu/publishers/google/models/gemini-3.1-flash-lite'
        response=self.session.post(f'https://{vertex_host("eu")}/v1/{resource}:countTokens',json={'contents':[{'role':'user','parts':[{'text':text}]}]},timeout=60,stream=True,allow_redirects=False)
        try:
            if response.status_code!=200:raise RuntimeError(f'Google countTokens HTTP {response.status_code}')
            return bounded_json(response)['totalTokens']
        finally:response.close()

    def predict(self,question,model,with_catalog=False,data=DATA):
        resource=model
        if not model.startswith('projects/'):
            location='eu'
            resource=f'projects/{self.project}/locations/eu/publishers/google/models/{model}'
        else:
            parts=model.split('/')
            if len(parts)!=6 or parts[2]!='locations' or parts[4]!='endpoints':raise ValueError('Usa endpoint real del trabajo, no URL o alias')
            if parts[1] not in (self.project,'653878673654'):raise ValueError('El endpoint pertenece a otro proyecto')
            location={'europe-west4':'eu','us-central1':'us'}.get(parts[3],parts[3])
            parts[3]=location;resource='/'.join(parts)
        body=payload(question,with_catalog,data)
        start=time.perf_counter()
        response=self.session.post(f'https://{vertex_host(location)}/v1/{resource}:generateContent',json=body,timeout=60,stream=True,allow_redirects=False)
        try:
            if response.status_code!=200:raise RuntimeError(f'Google HTTP {response.status_code}')
            reply=gemini_reply(bounded_json(response))
        finally:response.close()
        return {**reply,'latencia_s':time.perf_counter()-start,'model':resource,'ruta':'google','parametros_solicitados':body['generationConfig'],'con_catalogo':with_catalog,'coste_usd':None}
