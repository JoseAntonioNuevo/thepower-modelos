"""Límites locales para JSON HTTP; no repara, trunca ni reintenta respuestas."""
import json
import time

MAX_RESPONSE_BYTES = 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_PARTS = 64
MAX_MODELS = 256


def check_json_depth(text):
    depth = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise ValueError("JSON excede la profundidad permitida")
        elif char in "]}":
            depth -= 1


def bounded_json(response, max_bytes=MAX_RESPONSE_BYTES, max_seconds=120):
    """El llamador DEBE solicitar stream=True para impedir buffering previo.

    Se cuentan bytes descomprimidos y no se confía en Content-Length.
    El plazo se comprueba entre bloques; el timeout HTTP limita cada lectura.
    """
    try:
        length = response.headers.get("Content-Length")
        if length is not None and (int(length) < 0 or int(length) > max_bytes):
            raise ValueError("Respuesta HTTP excede el límite de bytes")
        body = bytearray()
        start = time.monotonic()
        for chunk in response.iter_content(chunk_size=8192):
            if time.monotonic() - start > max_seconds:
                raise ValueError("Respuesta HTTP excede el plazo de lectura")
            if len(body) + len(chunk) > max_bytes:
                raise ValueError("Respuesta HTTP excede el límite de bytes")
            body.extend(chunk)
        text = body.decode(json.detect_encoding(body))
        check_json_depth(text)
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Respuesta HTTP: se esperaba un objeto JSON")
        return data
    finally:
        response.close()


def objects(value, limit=MAX_PARTS):
    if not isinstance(value, list) or len(value) > limit or any(not isinstance(x, dict) for x in value):
        raise ValueError("Lista de respuesta mal formada o demasiado larga")
    return value


def model_ids(data):
    entries = objects(data.get("data", []), MAX_MODELS)
    ids = [entry.get("id") for entry in entries]
    if any(not isinstance(value, str) or not value or len(value) > 1024 for value in ids):
        raise ValueError("Identificador de modelo mal formado o demasiado largo")
    return ids


def model_text(value):
    # El transporte ya limita a 1 MiB. Conservar ese texto completo permite
    # marcar como inválido un resultado >16 KiB sin perder la evidencia.
    if value is not None and (not isinstance(value, str) or len(value) > MAX_RESPONSE_BYTES):
        raise ValueError("Texto del modelo mal formado o demasiado largo")
    return value


def joined_text(parts, key):
    texts = [model_text(part.get(key, "")) for part in objects(parts)]
    if any(text is None for text in texts) or sum(map(len, texts)) > MAX_RESPONSE_BYTES:
        raise ValueError("Texto del modelo mal formado o demasiado largo")
    return "".join(texts)


def first_choice(data):
    choices = objects(data.get("choices", []), 8)
    if not choices or not isinstance(choices[0].get("message"), dict):
        raise ValueError("Respuesta sin choice/message válido")
    return choices[0]
