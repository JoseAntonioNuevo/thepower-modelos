# GUÍA DE LECTURA · curso/remote.py
# Adaptadores de inferencia para OpenRouter, LM Studio y Google Cloud.
# Cada proveedor tiene su propia estructura de petición y de respuesta.
# Todos conservan texto original, parámetros solicitados y consumo disponible.
# La configuración se lee al llamar; importar este módulo no consulta una API.
# Google necesita un proyecto explícito del laboratorio; OpenRouter no realiza tuning.

"""Inferencia sin reparaciones, reintentos ni clientes creados al importar."""

# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
import json

# Lee configuración del entorno y ajusta opciones del proceso.
import os

# Mide duración y prepara fechas; perf_counter/monotonic sirven para intervalos, no fechas del calendario.
import time
from .common import ROOT, MAX_NEW_TOKENS, messages
from .http import (
    bounded_json,
    objects,
    model_ids,
    model_text,
    joined_text,
    first_choice,
)


# Lee .env si existe, conservando las variables ya definidas en el proceso.
# No crea clientes ni verifica credenciales; solo carga configuración.
def load_config():
    # Carga variables desde .env sin incluir claves en el código.
    from dotenv import load_dotenv

    # Lee .env sin sobrescribir configuración ya proporcionada por la terminal.
    load_dotenv(ROOT / ".env", override=False)


# Envia una incidencia por OpenRouter y conserva respuesta y metadatos.
# El modelo se elige explícitamente y las opciones del proveedor se registran.
def chat_openrouter(incident, model, fewshot=False):
    # Cliente HTTP; las peticiones se realizan al llamarlo, no por el simple import.
    import requests

    # Carga las variables necesarias justo antes de la petición, sin mostrarlas.
    load_config()
    # Clave leída desde el entorno; se usa en la cabecera de autenticación y no se incluye en resultados.
    key = os.getenv("OPENROUTER_API_KEY")
    # Validación previa: Falta OPENROUTER_API_KEY.
    if not key:
        raise ValueError("Falta OPENROUTER_API_KEY")
    # Cuerpo de la petición: instrucciones, entradas y parámetros con los nombres que exige esta API.
    payload = {
        "model": model,
        "messages": messages(incident, fewshot),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0,
        "usage": {"include": True},
    }
    # Solicita reasoning minimal para la ruta Google elegida a través de OpenRouter.
    if model.startswith("google/"):
        payload["reasoning"] = {"effort": "minimal"}
    # Solicita reasoning desactivado para la condición Claude de esta comparación.
    elif model.startswith("anthropic/"):
        payload["reasoning"] = {"enabled": False}
    # Exige que el proveedor soporte los parámetros solicitados; evita ignorarlos silenciosamente.
    payload["provider"] = {"require_parameters": True}
    # Inicio de la medida temporal; el tiempo de carga de pesos se mide por separado cuando corresponde.
    start = time.perf_counter()
    # Bearer autentica la petición; la clave no forma parte de los parámetros registrados.
    # Respuesta HTTP que se inspeccionará y cerrará después de leerla.
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        # Timeout separa conexión/lectura; stream=True permite lectura acotada y redirects quedan desactivados.
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "X-Title": "thePower S7 S8",
        },
        json=payload,
        timeout=(10, 60),
        stream=True,
        allow_redirects=False,
    )
    # Comprueba el estado HTTP antes de tratar el cuerpo como una respuesta válida del modelo.
    if response.status_code != 200:
        # Cierra el recurso abierto; libera socket/conexión aunque el paso anterior haya fallado.
        response.close()
        raise RuntimeError(
            f"OpenRouter HTTP {response.status_code}; no se reintenta ni cambia el modelo"
        )
    # Lee el JSON acotado y extrae el consumo usando el nombre propio de esta API.
    data = bounded_json(response)
    usage = data.get("usage") or {}
    # Primera alternativa válida de la respuesta del servicio.
    choice = first_choice(data)
    # Texto original que generó el modelo; no se repara antes de valorarlo o guardarlo.
    raw = model_text(choice["message"].get("content"))
    return {
        "respuesta_original": raw,
        "latencia_s": time.perf_counter() - start,
        # Tokens informados por el proveedor; los campos ausentes se conservan como None.
        "tokens_entrada": usage.get("prompt_tokens"),
        "tokens_salida": usage.get("completion_tokens"),
        # Solo registra importe real de usage.cost cuando está disponible, sin inventar una tarifa.
        "coste_usd": usage.get("cost"),
        "coste_metodo": "usage.cost de OpenRouter"
        if usage.get("cost") is not None
        else None,
        "model": data.get("model"),
        "provider": data.get("provider"),
        "request_id": data.get("id"),
        # Registra opciones solicitadas pero excluye el contenido de messages de este campo de metadatos.
        "parametros_solicitados": {
            k: v for k, v in payload.items() if k not in ("messages",)
        },
        "finish_reason": choice.get("finish_reason"),
    }


# Consulta LM Studio mediante REST nativa o interfaz compatible.
# Si falta modelo, solo lo deduce cuando el catálogo tiene exactamente una opción.
def chat_local(incident, model=None, fewshot=False, native=True):
    # Cliente HTTP; las peticiones se realizan al llamarlo, no por el simple import.
    import requests

    # Carga las variables necesarias justo antes de la petición, sin mostrarlas.
    load_config()
    # Normaliza la URL del servidor local para construir las rutas /models y /chat/completions.
    base = os.getenv("LOCAL_BASE_URL", "http://localhost:1234/v1").rstrip("/")
    # Prioriza el ID recibido por argumento y, si falta, el ID del entorno local.
    model = model or os.getenv("LOCAL_MODEL_ID")
    # Consulta catálogo solo si todavía no hay un modelo explícito.
    if model is None:
        # Respuesta HTTP del catálogo; enumera modelos disponibles sin generar una respuesta.
        listing = requests.get(
            base + "/models", timeout=10, stream=True, allow_redirects=False
        )
        try:
            # Comprueba el estado HTTP antes de tratar el cuerpo como una respuesta válida del modelo.
            if listing.status_code != 200:
                raise ValueError("Catálogo HTTP no disponible")
            # Convierte un estado HTTP de error en excepción antes de leer el catálogo.
            listing.raise_for_status()
            # Extrae únicamente identificadores válidos de la respuesta acotada.
            models = model_ids(bounded_json(listing))
        finally:
            # Cierra el recurso abierto; libera socket/conexión aunque el paso anterior haya fallado.
            listing.close()
        # Validación previa: Indica --modelo; el servidor no tiene exactamente un modelo.
        if len(models) != 1:
            raise ValueError(
                "Indica --modelo; el servidor no tiene exactamente un modelo"
            )
        # Selecciona la única opción del catálogo después de descartar cero o varias opciones.
        model = models[0]
    # Elige REST v1 de LM Studio; la otra rama mantiene el esquema compatible messages.
    if native:
        # Validación previa: Few-shot local: utiliza --local-api openai o el runner Qwen.
        if fewshot:
            raise ValueError(
                "Few-shot local: utiliza --local-api openai o el runner Qwen"
            )
        from .common import basic_prompt

        # Cuerpo de la petición: instrucciones, entradas y parámetros con los nombres que exige esta API.
        payload = {
            "model": model,
            "input": incident,
            "system_prompt": basic_prompt(),
            "reasoning": "off",
            "temperature": 0,
            "max_output_tokens": MAX_NEW_TOKENS,
            "store": False,
        }
        # REST nativa permite pedir reasoning off; store=False se envía como opción de esta API.
        # URL elegida para esta ruta de inferencia; cada API tiene una estructura propia.
        endpoint = base.removesuffix("/v1") + "/api/v1/chat"
    else:
        # Cuerpo de la petición: instrucciones, entradas y parámetros con los nombres que exige esta API.
        payload = {
            "model": model,
            "messages": messages(incident, fewshot),
            "temperature": 0,
            "max_tokens": MAX_NEW_TOKENS,
        }
        # URL elegida para esta ruta de inferencia; cada API tiene una estructura propia.
        endpoint = base + "/chat/completions"
    # Inicio de la medida temporal; el tiempo de carga de pesos se mide por separado cuando corresponde.
    start = time.perf_counter()
    # Respuesta HTTP que se inspeccionará y cerrará después de leerla.
    response = requests.post(
        endpoint, json=payload, timeout=(10, 90), stream=True, allow_redirects=False
    )
    # Comprueba el estado HTTP antes de tratar el cuerpo como una respuesta válida del modelo.
    if response.status_code != 200:
        # Cierra el recurso abierto; libera socket/conexión aunque el paso anterior haya fallado.
        response.close()
        raise RuntimeError(f"LM Studio HTTP {response.status_code}")
    # Lee el JSON acotado y extrae el consumo usando el nombre propio de esta API.
    data = bounded_json(response)
    usage = data.get("stats" if native else "usage") or {}
    # Texto original que generó el modelo; no se repara antes de valorarlo o guardarlo.
    raw = (
        joined_text(
            [x for x in objects(data.get("output", [])) if x.get("type") == "message"],
            "content",
        )
        # REST nativa devuelve bloques output; compatible devuelve choices[0].message.content.
        if native
        else model_text(first_choice(data)["message"].get("content"))
    )
    return {
        "respuesta_original": raw,
        "latencia_s": time.perf_counter() - start,
        "model": data.get("model", model),
        "runtime": "LM Studio",
        "tokens_entrada": usage.get("input_tokens" if native else "prompt_tokens"),
        "tokens_salida": usage.get(
            "total_output_tokens" if native else "completion_tokens"
        ),
        "api": "LM Studio REST v1" if native else "OpenAI compatible",
        "coste_usd": None,
        "coste_metodo": "Hardware/energía no medidos; sin tarifa de API",
        "parametros_solicitados": {
            k: v
            for k, v in payload.items()
            if k not in ("messages", "input", "system_prompt")
        },
    }


# Convierte una fila de S7 al formato systemInstruction + contents de Gemini.
# user contiene la incidencia y model la respuesta objetivo; ID/familia no son mensajes.
def gemini_record(row):
    """El JSONL conserva system y la respuesta objetivo; el ID va en un archivo separado."""
    from .common import basic_prompt

    # La entrada es role=user y la etiqueta/JSON de entrenamiento es role=model en el formato Gemini.
    return {
        "systemInstruction": {"parts": [{"text": basic_prompt()}]},
        "contents": [
            {"role": "user", "parts": [{"text": row["incidencia"]}]},
            # Conserva el JSON objetivo como texto para enseñar la misma tarea y no un objeto de configuración.
            {
                "role": "model",
                "parts": [{"text": json.dumps(row["objetivo"], ensure_ascii=False)}],
            },
        ],
    }


# Construye el cliente Google solo con proyecto del laboratorio explícito.
# location selecciona la región del servicio; la autenticación procede del entorno Google.
def google_client(location=None):
    # Carga las variables necesarias justo antes de la petición, sin mostrarlas.
    load_config()
    # Proyecto explícito del laboratorio que recibirá la llamada Google.
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    # Validación previa: Falta GOOGLE_CLOUD_PROJECT del laboratorio; no se usa el proyecto ADC por defecto.
    if not project:
        raise ValueError(
            "Falta GOOGLE_CLOUD_PROJECT del laboratorio; no se usa el proyecto ADC por defecto"
        )
    from google import genai
    from google.genai import types

    return genai.Client(
        vertexai=True,
        project=project,
        location=location or os.getenv("GEMINI_SERVING_LOCATION", "eu"),
        # Usa ubicación explícita o eu; es configuración de servicio, no la región del trabajo de tuning.
        http_options=types.HttpOptions(api_version="v1", timeout=60000),
    )


# Versión v1 y timeout del SDK expresado en milisegundos.


# Consulta generateContent autenticado con ADC en el proyecto y ubicación indicados.
# Devuelve solo bloques de respuesta, excluyendo los bloques marcados como pensamiento.
# El SDK/REST de inferencia no crea un trabajo de tuning.
def chat_google(incident, model):
    # REST en el transporte SDK: campo thinkingLevel admitido por la API actual,
    # aunque la versión docente del SDK sea anterior a su incorporación al tipo Python.
    # Carga las variables necesarias justo antes de la petición, sin mostrarlas.
    load_config()
    # Validación previa: Falta GOOGLE_CLOUD_PROJECT del laboratorio.
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        raise ValueError("Falta GOOGLE_CLOUD_PROJECT del laboratorio")
    from .common import basic_prompt

    # Inicio de la medida temporal; el tiempo de carga de pesos se mide por separado cuando corresponde.
    start = time.perf_counter()
    # Cuerpo de la petición: instrucciones, entradas y parámetros con los nombres que exige esta API.
    payload = {
        "systemInstruction": {"parts": [{"text": basic_prompt()}]},
        "contents": [{"role": "user", "parts": [{"text": incident}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": MAX_NEW_TOKENS,
            # Configura temperatura, límite de salida y thinking con el esquema de generateContent.
            "thinkingConfig": {"thinkingLevel": "MINIMAL"},
        },
    }
    # Uso de REST autorizado por ADC evita depender de tipos privados del SDK.
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    # Obtiene ADC con el scope cloud-platform; la identidad real procede del entorno Google autorizado.
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    # Proyecto explícito del laboratorio que recibirá la llamada Google.
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.getenv("GEMINI_SERVING_LOCATION", "eu")
    # Nombre completo del recurso Google: modelo publicado o recurso ajustado.
    resource = (
        model
        if model.startswith("projects/")
        else f"projects/{project}/locations/{location}/publishers/google/models/{model}"
    )
    # Dirección de la petición o del servidor local que se va a consultar.
    url = f"https://{location}-aiplatform.googleapis.com/v1/{resource}:generateContent"
    # Sesión HTTP que aplica la autenticación Google a las peticiones.
    session = AuthorizedSession(credentials)
    try:
        # Respuesta HTTP que se inspeccionará y cerrará después de leerla.
        response = session.post(
            url, json=payload, timeout=60, stream=True, allow_redirects=False
        )
        # Comprueba el estado HTTP antes de tratar el cuerpo como una respuesta válida del modelo.
        if response.status_code != 200:
            # Cierra el recurso abierto; libera socket/conexión aunque el paso anterior haya fallado.
            response.close()
            raise RuntimeError(f"Google HTTP {response.status_code}")
        # Lee el JSON acotado y extrae el consumo usando el nombre propio de esta API.
        data = bounded_json(response)
        usage = data.get("usageMetadata") or {}
        # Alternativas de generación del proveedor, comprobadas como lista acotada.
        candidates = objects(data.get("candidates", []), 8)
        # Validación previa: Google no devolvió ningún candidato.
        if not candidates:
            raise ValueError("Google no devolvió ningún candidato")
        # Contenido de la primera alternativa; se comprobará antes de extraer bloques de texto.
        content = candidates[0].get("content", {})
        # Validación previa: Contenido Google mal formado.
        if not isinstance(content, dict):
            raise ValueError("Contenido Google mal formado")
        # Bloques de contenido que se recorren para reconstruir la respuesta.
        parts = objects(content.get("parts", []))
        # Texto original que generó el modelo; no se repara antes de valorarlo o guardarlo.
        raw = joined_text([part for part in parts if not part.get("thought")], "text")
        return {
            "respuesta_original": raw,
            "latencia_s": time.perf_counter() - start,
            "tokens_entrada": usage.get("promptTokenCount"),
            "tokens_salida": usage.get("candidatesTokenCount"),
            "coste_usd": None,
            "coste_metodo": "La API no devuelve importe; consultar facturación/tarifa por separado",
            "model": resource,
            "provider": "Google Cloud",
            "parametros_solicitados": payload["generationConfig"],
        }
    finally:
        # Cierra el recurso abierto; libera socket/conexión aunque el paso anterior haya fallado.
        session.close()
