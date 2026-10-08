"""Inferencia sin reparaciones, reintentos ni clientes creados al importar."""
import json
import os
import time
from .common import ROOT, MAX_NEW_TOKENS, messages
from .http import bounded_json, objects, model_ids, model_text, joined_text, first_choice


def load_config():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)


def chat_openrouter(incident, model, fewshot=False):
    import requests
    load_config()
    key=os.getenv("OPENROUTER_API_KEY")
    if not key: raise ValueError("Falta OPENROUTER_API_KEY")
    payload={"model":model,"messages":messages(incident,fewshot),"max_tokens":MAX_NEW_TOKENS,
             "temperature":0,"usage":{"include":True}}
    if model.startswith("google/"): payload["reasoning"]={"effort":"minimal"}
    elif model.startswith("anthropic/"): payload["reasoning"]={"enabled":False}
    payload["provider"]={"require_parameters":True}
    start=time.perf_counter()
    response=requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization":"Bearer "+key,"Content-Type":"application/json","X-Title":"thePower S7 S8"},
        json=payload,timeout=(10,60),stream=True,allow_redirects=False)
    if response.status_code!=200:
        response.close()
        raise RuntimeError(f"OpenRouter HTTP {response.status_code}; no se reintenta ni cambia el modelo")
    data=bounded_json(response);usage=data.get("usage") or {}
    choice=first_choice(data)
    raw=model_text(choice["message"].get("content"))
    return {"respuesta_original":raw,"latencia_s":time.perf_counter()-start,
            "tokens_entrada":usage.get("prompt_tokens"),"tokens_salida":usage.get("completion_tokens"),
            "coste_usd":usage.get("cost"),"coste_metodo":"usage.cost de OpenRouter" if usage.get("cost") is not None else None,
            "model":data.get("model"),"provider":data.get("provider"),"request_id":data.get("id"),
            "parametros_solicitados":{k:v for k,v in payload.items() if k not in ("messages",)},
            "finish_reason":choice.get("finish_reason")}


def chat_local(incident, model=None, fewshot=False, native=True):
    import requests
    load_config()
    base=os.getenv("LOCAL_BASE_URL","http://localhost:1234/v1").rstrip("/")
    model = model or os.getenv("LOCAL_MODEL_ID")
    if model is None:
        listing=requests.get(base+"/models",timeout=10,stream=True,allow_redirects=False)
        try:
            if listing.status_code!=200:raise ValueError("Catálogo HTTP no disponible")
            listing.raise_for_status()
            models=model_ids(bounded_json(listing))
        finally:
            listing.close()
        if len(models)!=1: raise ValueError("Indica --modelo; el servidor no tiene exactamente un modelo")
        model=models[0]
    if native:
        if fewshot: raise ValueError("Few-shot local: utiliza --local-api openai o el runner Qwen")
        from .common import basic_prompt
        payload={"model":model,"input":incident,"system_prompt":basic_prompt(),"reasoning":"off",
                 "temperature":0,"max_output_tokens":MAX_NEW_TOKENS,"store":False}
        endpoint=base.removesuffix("/v1")+"/api/v1/chat"
    else:
        payload={"model":model,"messages":messages(incident,fewshot),"temperature":0,"max_tokens":MAX_NEW_TOKENS}
        endpoint=base+"/chat/completions"
    start=time.perf_counter()
    response=requests.post(endpoint,json=payload,timeout=(10,90),stream=True,allow_redirects=False)
    if response.status_code!=200:
        response.close()
        raise RuntimeError(f"LM Studio HTTP {response.status_code}")
    data=bounded_json(response);usage=data.get("stats" if native else "usage") or {}
    raw=(joined_text([x for x in objects(data.get("output",[])) if x.get("type")=="message"],"content")
         if native else model_text(first_choice(data)["message"].get("content")))
    return {"respuesta_original":raw,
            "latencia_s":time.perf_counter()-start,"model":data.get("model",model),"runtime":"LM Studio",
            "tokens_entrada":usage.get("input_tokens" if native else "prompt_tokens"),
            "tokens_salida":usage.get("total_output_tokens" if native else "completion_tokens"),
            "api":"LM Studio REST v1" if native else "OpenAI compatible",
            "coste_usd":None,"coste_metodo":"Hardware/energía no medidos; sin tarifa de API",
            "parametros_solicitados":{k:v for k,v in payload.items() if k not in ("messages","input","system_prompt")}}


def gemini_record(row):
    """El JSONL conserva system y la respuesta objetivo; el ID va en un archivo separado."""
    from .common import basic_prompt
    return {"systemInstruction":{"parts":[{"text":basic_prompt()}]},
            "contents":[{"role":"user","parts":[{"text":row["incidencia"]}]},
                        {"role":"model","parts":[{"text":json.dumps(row["objetivo"],ensure_ascii=False)}]}]}


def google_client(location=None):
    load_config()
    project=os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project: raise ValueError("Falta GOOGLE_CLOUD_PROJECT del laboratorio; no se usa el proyecto ADC por defecto")
    from google import genai
    from google.genai import types
    return genai.Client(vertexai=True,project=project,
        location=location or os.getenv("GEMINI_SERVING_LOCATION","eu"),
        http_options=types.HttpOptions(api_version="v1",timeout=60000))


def chat_google(incident, model):
    # REST en el transporte SDK: campo thinkingLevel admitido por la API actual,
    # aunque la versión docente del SDK sea anterior a su incorporación al tipo Python.
    load_config()
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        raise ValueError("Falta GOOGLE_CLOUD_PROJECT del laboratorio")
    from .common import basic_prompt
    start=time.perf_counter()
    payload={"systemInstruction":{"parts":[{"text":basic_prompt()}]},
             "contents":[{"role":"user","parts":[{"text":incident}]}],
             "generationConfig":{"temperature":0,"maxOutputTokens":MAX_NEW_TOKENS,
                                  "thinkingConfig":{"thinkingLevel":"MINIMAL"}}}
    # Uso de REST autorizado por ADC evita depender de tipos privados del SDK.
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    credentials,_=google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    project=os.environ["GOOGLE_CLOUD_PROJECT"];location=os.getenv("GEMINI_SERVING_LOCATION","eu")
    resource=model if model.startswith("projects/") else f"projects/{project}/locations/{location}/publishers/google/models/{model}"
    url=f"https://{location}-aiplatform.googleapis.com/v1/{resource}:generateContent"
    session=AuthorizedSession(credentials)
    try:
        response=session.post(url,json=payload,timeout=60,stream=True,allow_redirects=False)
        if response.status_code!=200:
            response.close()
            raise RuntimeError(f"Google HTTP {response.status_code}")
        data=bounded_json(response);usage=data.get("usageMetadata") or {}
        candidates=objects(data.get("candidates",[]),8)
        if not candidates:raise ValueError("Google no devolvió ningún candidato")
        content=candidates[0].get("content",{})
        if not isinstance(content,dict):raise ValueError("Contenido Google mal formado")
        parts=objects(content.get("parts",[]))
        raw=joined_text([part for part in parts if not part.get("thought")],"text")
        return {"respuesta_original":raw,"latencia_s":time.perf_counter()-start,
                "tokens_entrada":usage.get("promptTokenCount"),"tokens_salida":usage.get("candidatesTokenCount"),
                "coste_usd":None,"coste_metodo":"La API no devuelve importe; consultar facturación/tarifa por separado",
                "model":resource,"provider":"Google Cloud","parametros_solicitados":payload["generationConfig"]}
    finally:
        session.close()
