# GUÍA DE LECTURA · curso/remote.py
# Adaptadores de inferencia remota y local, independientes del proveedor.
# chat() lee LLM_PROVIDER y LLM_MODEL del .env y llama al adaptador de esa ruta.
# Rutas: Anthropic (SDK oficial), OpenAI, OpenRouter, cualquier API compatible con
# Chat Completions, Gemini API con clave, LM Studio y Vertex AI en Google Cloud.
# Cada proveedor tiene su propia estructura de petición y de respuesta.
# Todos conservan texto original, parámetros solicitados y consumo disponible.
# La configuración se lee al llamar; importar este módulo no consulta una API.
# Google Cloud necesita un proyecto explícito del laboratorio; ninguna ruta realiza tuning.

"""Inferencia sin reparaciones, reintentos ni clientes creados al importar."""

# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
import json

# Lee configuración del entorno y ajusta opciones del proceso.
import os

# Mide duración y prepara fechas; perf_counter/monotonic sirven para intervalos, no fechas del calendario.
import time

# Codifica el ID del modelo dentro de una URL y analiza URLs base configuradas en .env.
from urllib.parse import quote, urlsplit
from .common import ROOT, MAX_NEW_TOKENS, messages
from .http import (
    bounded_json,
    objects,
    model_ids,
    model_text,
    joined_text,
    first_choice,
)


# Rutas que admite chat(); la clave es el valor de LLM_PROVIDER o de --proveedor.
PROVEEDORES = {
    "anthropic": "Messages API de Anthropic con el SDK oficial",
    "openai": "Chat Completions de OpenAI",
    "openrouter": "OpenRouter, interfaz común para muchos proveedores",
    "compatible": "Cualquier API compatible con Chat Completions (COMPAT_BASE_URL)",
    "gemini": "Gemini API de Google con clave (GEMINI_API_KEY)",
    "local": "LM Studio en tu equipo",
    "google": "Vertex AI en el proyecto Google Cloud del laboratorio",
}
# Ruta del curso cuando .env no indica otra: Claude Haiku 5.5 por la API de Anthropic.
PROVEEDOR_POR_DEFECTO = "anthropic"
MODELO_POR_DEFECTO = "claude-haiku-5-5"
# Campos que identifican la petición; LLM_EXTRA_BODY no puede sustituirlos.
CAMPOS_PROTEGIDOS = {
    "model",
    "messages",
    "system",
    "contents",
    "systemInstruction",
    "stream",
    "input",
    "system_prompt",
}


# Lee .env si existe, conservando las variables ya definidas en el proceso.
# No crea clientes ni verifica credenciales; solo carga configuración.
def load_config():
    # Carga variables desde .env sin incluir claves en el código.
    from dotenv import load_dotenv

    # Lee .env sin sobrescribir configuración ya proporcionada por la terminal.
    load_dotenv(ROOT / ".env", override=False)


# Proveedor configurado en .env, o el del curso si no hay ninguno.
def configured_provider():
    # Carga las variables necesarias justo antes de decidir la ruta.
    load_config()
    return (os.getenv("LLM_PROVIDER") or PROVEEDOR_POR_DEFECTO).strip().lower()


# Modelo de LLM_MODEL; sin él, el del curso solo si la ruta es la del curso.
def configured_model():
    # Proveedor de .env; LLM_MODEL pertenece a esa ruta.
    provider = configured_provider()
    return os.getenv("LLM_MODEL") or (
        MODELO_POR_DEFECTO if provider == PROVEEDOR_POR_DEFECTO else None
    )


# Decide proveedor y modelo: argumentos explícitos primero y, si faltan, .env.
# LLM_MODEL solo se aplica al proveedor de LLM_PROVIDER para no mezclar IDs de rutas distintas.
def route(provider=None, model=None):
    # Proveedor que el usuario dejó configurado en .env.
    configured = configured_provider()
    # Prioriza --proveedor; si falta, usa la ruta de .env.
    provider = (provider or configured).strip().lower()
    # Validación previa: rutas desconocidas fallan antes de construir una petición.
    if provider not in PROVEEDORES:
        raise ValueError(
            f"Proveedor desconocido {provider!r}; usa uno de: {', '.join(PROVEEDORES)}"
        )
    # Completa el modelo desde .env solo cuando la ruta coincide con la configurada.
    if not model and provider == configured:
        model = configured_model()
    # Validación previa: salvo LM Studio, que puede descubrirlo, el modelo debe ser explícito.
    if not model and provider != "local":
        raise ValueError(f"Indica el modelo de {provider} con --modelo o LLM_MODEL")
    return provider, model


# Credencial o configuración mínima de cada ruta, comprobada antes de un lote.
# Evita registrar 30 errores iguales cuando falta la clave; no muestra valores.
def require_credentials(provider):
    # Carga las variables necesarias justo antes de comprobarlas.
    load_config()
    # Variable que debe existir para cada ruta; local no necesita ninguna.
    needed = {
        "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        "openai": ("OPENAI_API_KEY",),
        "openrouter": ("OPENROUTER_API_KEY",),
        "gemini": ("GEMINI_API_KEY",),
        "compatible": ("COMPAT_BASE_URL",),
        "google": ("GOOGLE_CLOUD_PROJECT",),
    }.get(provider, ())
    # Validación previa: basta con una de las variables admitidas para esa ruta.
    if needed and not any(os.getenv(name) for name in needed):
        raise ValueError(f"Falta {needed[0]} para la ruta {provider}")


# Ajustes opcionales del .env para la ruta configurada: temperatura y campos extra.
# Devuelve ("defecto", {}) cuando la ruta no es la de .env, para no aplicarle opciones ajenas.
def route_settings(provider):
    # Solo la ruta de LLM_PROVIDER recibe LLM_TEMPERATURE y LLM_EXTRA_BODY.
    if provider != configured_provider():
        return "defecto", {}
    # Texto de LLM_TEMPERATURE: vacío conserva el valor del adaptador; "omitir" no la envía.
    raw = (os.getenv("LLM_TEMPERATURE") or "").strip().lower()
    if not raw:
        temperature = "defecto"
    elif raw in ("omitir", "none", "null"):
        temperature = None
    else:
        temperature = float(raw)
    # Campos adicionales en JSON, por ejemplo reasoning_effort o thinkingConfig de un proveedor.
    text = (os.getenv("LLM_EXTRA_BODY") or "").strip()
    extra = json.loads(text) if text else {}
    # Validación previa: debe ser un objeto JSON y no puede reemplazar modelo ni mensajes.
    if not isinstance(extra, dict):
        raise ValueError("LLM_EXTRA_BODY debe ser un objeto JSON")
    blocked = CAMPOS_PROTEGIDOS & set(extra)
    if blocked:
        raise ValueError(f"LLM_EXTRA_BODY no puede cambiar {sorted(blocked)}")
    return temperature, extra


# Mezcla diccionarios anidados: los valores de extra sustituyen o completan a los de base.
# Un valor null en extra elimina ese campo de la petición.
def merged(base, extra):
    # Copia superficial para no modificar el diccionario recibido.
    result = dict(base)
    for key, value in extra.items():
        # null en LLM_EXTRA_BODY retira un campo que el adaptador enviaría por defecto.
        if value is None:
            result.pop(key, None)
        # Si ambos son objetos, combina sus claves en lugar de reemplazar el objeto entero.
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merged(result[key], value)
        else:
            result[key] = value
    return result


# Coste calculado con la tarifa declarada en .env para el modelo de LLM_MODEL.
# Devuelve (None, motivo) si no hay tarifa o si el modelo usado es otro.
def tariff_cost(provider, model, tokens_in, tokens_out):
    # Tarifas en dólares por millón de tokens, escritas por el usuario en .env.
    price_in = os.getenv("LLM_PRECIO_ENTRADA_USD_MTOK")
    price_out = os.getenv("LLM_PRECIO_SALIDA_USD_MTOK")
    # Solo aplica la tarifa a la ruta y el modelo para los que se declaró.
    same_route = provider == configured_provider() and model == configured_model()
    # Sin tarifa, ruta distinta o tokens desconocidos: el coste queda sin medir.
    if not (price_in and price_out and same_route) or None in (tokens_in, tokens_out):
        return None, "Sin tarifa aplicable en .env; calcular tokens × precio publicado"
    # Importe en dólares: tokens de entrada y de salida por su precio por millón.
    cost = (tokens_in * float(price_in) + tokens_out * float(price_out)) / 1_000_000
    return cost, "tokens × tarifa de .env (LLM_PRECIO_*); verificar el precio vigente"


# Punto de entrada común: decide la ruta y llama a su adaptador.
# fewshot añade los cuatro ejemplos fijados de train; no entrena ningún modelo.
def chat(incident, provider=None, model=None, fewshot=False):
    # Proveedor y modelo resueltos desde argumentos o .env.
    provider, model = route(provider, model)
    # Cada ruta conserva su propio formato de petición y respuesta.
    if provider == "anthropic":
        return chat_anthropic(incident, model, fewshot)
    if provider == "openai":
        return chat_openai(incident, model, fewshot)
    if provider == "openrouter":
        return chat_openrouter(incident, model, fewshot)
    if provider == "compatible":
        return chat_compatible(incident, model, fewshot)
    if provider == "gemini":
        return chat_gemini(incident, model, fewshot)
    # LM Studio usa su API nativa salvo con few-shot, que necesita la interfaz compatible.
    if provider == "local":
        return chat_local(incident, model, fewshot, native=not fewshot)
    # Vertex AI: base o recurso ajustado del laboratorio; la condición del ejercicio es sin few-shot.
    if fewshot:
        raise ValueError(
            "La ruta google del ejercicio usa el prompt básico, sin few-shot"
        )
    return chat_google(incident, model)


# Envía una incidencia a Claude por la Messages API con el SDK oficial de Anthropic.
# max_retries=0 conserva el protocolo: un fallo cuenta como error, sin reintentos.
def chat_anthropic(incident, model, fewshot=False):
    # Carga las variables necesarias justo antes de la petición, sin mostrarlas.
    load_config()
    # Validación previa: sin credencial no se construye el cliente.
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise ValueError("Falta ANTHROPIC_API_KEY")
    # SDK oficial; se importa al llamar para que importar curso.remote no conecte.
    import anthropic

    # Mensajes comunes: el primero es system y el resto alterna user/assistant y acaba en user.
    conversation = messages(incident, fewshot)
    # Claude recibe las instrucciones en el campo system, fuera de messages.
    request = {
        "model": model,
        "max_tokens": MAX_NEW_TOKENS,
        "system": conversation[0]["content"],
        "messages": conversation[1:],
        # Sin razonamiento extendido: los 180 tokens de salida quedan para la respuesta.
        "thinking": {"type": "disabled"},
    }
    # Opciones de .env: temperatura y campos extra (por ejemplo, otro thinking u output_config).
    temperature, extra = route_settings("anthropic")
    # Haiku 5.5 rechaza temperaturas distintas de la predeterminada: solo se envía si .env la fija.
    if temperature not in ("defecto", None):
        request["temperature"] = temperature
    # thinking puede sustituirse o retirarse desde .env; el resto viaja como extra_body.
    if "thinking" in extra:
        if extra["thinking"] is None:
            request.pop("thinking")
        else:
            request["thinking"] = extra["thinking"]
    extra_body = {k: v for k, v in extra.items() if k != "thinking"}
    # Cliente sin reintentos automáticos y con un tiempo máximo de un minuto.
    client = anthropic.Anthropic(max_retries=0, timeout=60.0)
    # Inicio de la medida temporal alrededor de la llamada completa.
    start = time.perf_counter()
    try:
        message = client.messages.create(**request, extra_body=extra_body or None)
    # Errores HTTP del servicio: se registra el estado y no se reintenta ni se cambia de modelo.
    except anthropic.APIStatusError as exc:
        raise RuntimeError(
            f"Anthropic HTTP {exc.status_code}; no se reintenta ni cambia el modelo"
        ) from None
    # Fallo de red o tiempo agotado antes de recibir respuesta.
    except anthropic.APIConnectionError:
        raise RuntimeError("Anthropic sin respuesta; no se reintenta") from None
    finally:
        # Cierra las conexiones del cliente aunque la llamada haya fallado.
        client.close()
    # Texto original: solo bloques de tipo text, en orden; los bloques thinking no son respuesta.
    raw = "".join(block.text for block in message.content if block.type == "text")
    # Tokens informados por Anthropic para esta petición.
    tokens_in, tokens_out = message.usage.input_tokens, message.usage.output_tokens
    # La API no devuelve importe: se calcula con la tarifa declarada en .env, si existe.
    cost, method = tariff_cost("anthropic", model, tokens_in, tokens_out)
    return {
        "respuesta_original": raw,
        "latencia_s": time.perf_counter() - start,
        "tokens_entrada": tokens_in,
        "tokens_salida": tokens_out,
        "coste_usd": cost,
        "coste_metodo": method,
        "model": message.model,
        "provider": "Anthropic",
        "ruta": "anthropic",
        # request-id de la cabecera HTTP e ID del mensaje, útiles para trazar una llamada.
        "request_id": message._request_id,
        "message_id": message.id,
        # Registra opciones solicitadas sin las instrucciones ni los mensajes.
        "parametros_solicitados": {
            **{k: v for k, v in request.items() if k not in ("system", "messages")},
            **extra_body,
        },
        # end_turn, max_tokens o refusal; refusal deja el texto vacío y cuenta como error.
        "finish_reason": message.stop_reason,
    }


# Comprueba la URL base de una API configurada en .env antes de enviarle una clave.
# https es obligatorio salvo para servidores del propio equipo (Ollama, vLLM, LM Studio).
def checked_base_url(name, base_url):
    # Partes de la URL: esquema y servidor deciden si es aceptable.
    parts = urlsplit(base_url or "")
    # Validación previa: una URL sin esquema o sin servidor no se puede consultar.
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"Falta {name} con http(s)://servidor/…")
    # Las claves solo viajan sin cifrar hacia servidores de tu propio equipo.
    if parts.scheme == "http" and parts.hostname not in (
        "localhost",
        "127.0.0.1",
        "::1",
    ):
        raise ValueError(f"{name} remota debe usar https")
    return base_url


# Petición Chat Completions común a OpenAI, OpenRouter y APIs compatibles.
# token_field cambia según el proveedor: max_tokens o max_completion_tokens.
def chat_completions(
    incident,
    model,
    fewshot,
    *,
    provider,
    base_url,
    key,
    token_field="max_tokens",
    temperature=0,
    defaults=None,
    headers=None,
):
    # Cliente HTTP; las peticiones se realizan al llamarlo, no por el simple import.
    import requests

    # Cuerpo de la petición: instrucciones, entradas y parámetros con los nombres que exige esta API.
    payload = {
        "model": model,
        "messages": messages(incident, fewshot),
        token_field: MAX_NEW_TOKENS,
        **(defaults or {}),
    }
    # Opciones de .env: temperatura y campos extra, solo para la ruta configurada.
    env_temperature, extra = route_settings(provider)
    # Temperatura del adaptador salvo que .env la cambie u ordene omitirla.
    temperature = temperature if env_temperature == "defecto" else env_temperature
    if temperature is not None:
        payload["temperature"] = temperature
    # Combina los campos extra de .env con los del adaptador.
    payload = merged(payload, extra)
    # Cabeceras: autenticación Bearer solo si hay clave; los servidores locales pueden no pedirla.
    sent_headers = {"Content-Type": "application/json", **(headers or {})}
    if key:
        sent_headers["Authorization"] = "Bearer " + key
    # Inicio de la medida temporal alrededor de la llamada completa.
    start = time.perf_counter()
    # Timeout separa conexión/lectura; stream=True permite lectura acotada y redirects quedan desactivados.
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers=sent_headers,
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
            f"{provider} HTTP {response.status_code}; no se reintenta ni cambia el modelo"
        )
    # Lee el JSON acotado y extrae el consumo usando el nombre propio de esta API.
    data = bounded_json(response)
    usage = data.get("usage") or {}
    # Primera alternativa válida de la respuesta del servicio.
    choice = first_choice(data)
    # Texto original que generó el modelo; no se repara antes de valorarlo o guardarlo.
    raw = model_text(choice["message"].get("content"))
    tokens_in, tokens_out = usage.get("prompt_tokens"), usage.get("completion_tokens")
    # OpenRouter informa el importe en usage.cost; en otras rutas se usa la tarifa de .env.
    if usage.get("cost") is not None:
        cost, method = usage.get("cost"), f"usage.cost de {provider}"
    else:
        cost, method = tariff_cost(provider, model, tokens_in, tokens_out)
    return {
        "respuesta_original": raw,
        "latencia_s": time.perf_counter() - start,
        # Tokens informados por el proveedor; los campos ausentes se conservan como None.
        "tokens_entrada": tokens_in,
        "tokens_salida": tokens_out,
        "coste_usd": cost,
        "coste_metodo": method,
        "model": data.get("model"),
        # OpenRouter indica qué proveedor sirvió la petición; si no, se registra la ruta.
        "provider": data.get("provider") or provider,
        "ruta": provider,
        "request_id": data.get("id"),
        # Registra opciones solicitadas pero excluye el contenido de messages de este campo de metadatos.
        "parametros_solicitados": {
            k: v for k, v in payload.items() if k not in ("messages",)
        },
        "finish_reason": choice.get("finish_reason"),
    }


# OpenAI: Chat Completions con max_completion_tokens y sin temperatura por defecto,
# porque sus modelos de razonamiento rechazan valores distintos del predeterminado.
def chat_openai(incident, model, fewshot=False):
    # Carga las variables necesarias justo antes de la petición, sin mostrarlas.
    load_config()
    # Clave leída desde el entorno; se usa en la cabecera de autenticación y no se incluye en resultados.
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Falta OPENAI_API_KEY")
    return chat_completions(
        incident,
        model,
        fewshot,
        provider="openai",
        base_url=checked_base_url(
            "OPENAI_BASE_URL",
            os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        ),
        key=key,
        token_field="max_completion_tokens",
        temperature=None,
    )


# Envía una incidencia por OpenRouter y conserva respuesta y metadatos.
# El modelo se elige explícitamente y las opciones del proveedor se registran.
def chat_openrouter(incident, model, fewshot=False):
    # Carga las variables necesarias justo antes de la petición, sin mostrarlas.
    load_config()
    # Clave leída desde el entorno; se usa en la cabecera de autenticación y no se incluye en resultados.
    key = os.getenv("OPENROUTER_API_KEY")
    # Validación previa: Falta OPENROUTER_API_KEY.
    if not key:
        raise ValueError("Falta OPENROUTER_API_KEY")
    # usage.include pide el importe; require_parameters evita proveedores que ignoren parámetros.
    defaults = {"usage": {"include": True}, "provider": {"require_parameters": True}}
    # Razonamiento mínimo para Gemini y desactivado para Claude, como en el protocolo anterior.
    if model.startswith("google/"):
        defaults["reasoning"] = {"effort": "minimal"}
    elif model.startswith("anthropic/"):
        defaults["reasoning"] = {"enabled": False}
    return chat_completions(
        incident,
        model,
        fewshot,
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        key=key,
        defaults=defaults,
        headers={"X-Title": "thePower S7 S8"},
    )


# Cualquier API compatible con Chat Completions: Groq, Mistral, DeepSeek, Together,
# Ollama, vLLM… La URL base y la clave salen de COMPAT_BASE_URL y COMPAT_API_KEY.
def chat_compatible(incident, model, fewshot=False):
    # Carga las variables necesarias justo antes de la petición, sin mostrarlas.
    load_config()
    return chat_completions(
        incident,
        model,
        fewshot,
        provider="compatible",
        base_url=checked_base_url("COMPAT_BASE_URL", os.getenv("COMPAT_BASE_URL")),
        key=os.getenv("COMPAT_API_KEY"),
        token_field=os.getenv("COMPAT_TOKEN_FIELD") or "max_tokens",
    )


# Extrae texto, tokens y metadatos de una respuesta generateContent de Gemini.
# Excluye los bloques marcados como pensamiento; no repara el texto.
def gemini_reply(data):
    # Consumo en el formato propio de Google.
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
    return {
        # Texto original que generó el modelo; no se repara antes de valorarlo o guardarlo.
        "respuesta_original": joined_text(
            [part for part in parts if not part.get("thought")], "text"
        ),
        "tokens_entrada": usage.get("promptTokenCount"),
        "tokens_salida": usage.get("candidatesTokenCount"),
        "finish_reason": candidates[0].get("finishReason"),
    }


# Cuerpo generateContent: instrucciones, conversación y configuración de generación.
# few-shot convierte assistant en role=model, que es el nombre que usa Gemini.
def gemini_payload(incident, fewshot=False):
    # Mensajes comunes: el primero es system y el resto alterna user/assistant y acaba en user.
    conversation = messages(incident, fewshot)
    return {
        "systemInstruction": {"parts": [{"text": conversation[0]["content"]}]},
        "contents": [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in conversation[1:]
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": MAX_NEW_TOKENS,
            # Pensamiento mínimo, igual que en la ruta Vertex del ejercicio.
            "thinkingConfig": {"thinkingLevel": "MINIMAL"},
        },
    }


# Gemini API de Google con clave (Google AI Studio), sin proyecto de Google Cloud.
# Para el ajuste gestionado y su modelo resultante se usa chat_google (Vertex AI).
def chat_gemini(incident, model, fewshot=False):
    # Cliente HTTP; las peticiones se realizan al llamarlo, no por el simple import.
    import requests

    # Carga las variables necesarias justo antes de la petición, sin mostrarlas.
    load_config()
    # Clave leída desde el entorno; viaja en la cabecera x-goog-api-key, nunca en la URL.
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Falta GEMINI_API_KEY")
    # El ID puede llegar como models/{id}; se codifica para que no altere la ruta de la URL.
    name = quote(model.removeprefix("models/"), safe="-._")
    # Cuerpo común y opciones de .env (por ejemplo, otro thinkingLevel para modelos Pro).
    payload = gemini_payload(incident, fewshot)
    temperature, extra = route_settings("gemini")
    if temperature is None:
        payload["generationConfig"].pop("temperature")
    elif temperature != "defecto":
        payload["generationConfig"]["temperature"] = temperature
    payload = merged(payload, extra)
    # Inicio de la medida temporal alrededor de la llamada completa.
    start = time.perf_counter()
    # Respuesta HTTP que se inspeccionará y cerrará después de leerla.
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{name}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
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
            f"Gemini HTTP {response.status_code}; no se reintenta ni cambia el modelo"
        )
    # Lee el JSON acotado y separa texto, tokens y motivo de parada.
    data = bounded_json(response)
    reply = gemini_reply(data)
    # La API no devuelve importe: se calcula con la tarifa declarada en .env, si existe.
    cost, method = tariff_cost(
        "gemini", model, reply["tokens_entrada"], reply["tokens_salida"]
    )
    return {
        **reply,
        "latencia_s": time.perf_counter() - start,
        "coste_usd": cost,
        "coste_metodo": method,
        "model": data.get("modelVersion") or model,
        "provider": "Google · Gemini API",
        "ruta": "gemini",
        "request_id": data.get("responseId"),
        "parametros_solicitados": {
            k: v
            for k, v in payload.items()
            if k not in ("contents", "systemInstruction")
        },
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
    # Inicio de la medida temporal; el tiempo de carga de pesos se mide por separado cuando corresponde.
    start = time.perf_counter()
    # Cuerpo de la petición: el mismo que la Gemini API, con temperatura 0, 180 tokens y thinking MINIMAL.
    payload = gemini_payload(incident)
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
        # Lee el JSON acotado y separa texto, tokens y motivo de parada.
        reply = gemini_reply(bounded_json(response))
        return {
            **reply,
            "latencia_s": time.perf_counter() - start,
            "coste_usd": None,
            "coste_metodo": "La API no devuelve importe; consultar facturación/tarifa por separado",
            "model": resource,
            "provider": "Google Cloud",
            "ruta": "google",
            "parametros_solicitados": payload["generationConfig"],
        }
    finally:
        # Cierra el recurso abierto; libera socket/conexión aunque el paso anterior haya fallado.
        session.close()
