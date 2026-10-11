# GUÍA DE LECTURA · curso/http.py
# Lectura acotada de respuestas HTTP procedentes de servicios externos.
# stream=True permite inspeccionar bloques antes de acumular el cuerpo completo.
# Aquí limitamos bytes y profundidad y comprobamos la estructura esperada.
# Estos controles rechazan respuestas inválidas; no reparan el texto del modelo.
# Cerrar la respuesta libera su conexión, también cuando aparece un error.

"""Límites locales para JSON HTTP; no repara, trunca ni reintenta respuestas."""

# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
import json

# Mide duración y prepara fechas; perf_counter/monotonic sirven para intervalos, no fechas del calendario.
import time

# Máximo del transporte: 1 MiB de datos recibidos y descomprimidos.
MAX_RESPONSE_BYTES = 1024 * 1024
# Límite de anidamiento para no aceptar estructuras arbitrariamente profundas.
MAX_JSON_DEPTH = 32
# Admite como máximo 64 bloques de texto/objetos por respuesta.
MAX_PARTS = 64
# Limita el catálogo a 256 entradas antes de extraer sus IDs.
MAX_MODELS = 256


# Cuenta anidamiento de objetos/listas fuera de cadenas entre comillas.
# Las llaves escritas dentro de un texto no cuentan como estructura JSON.
# Solo limita profundidad: json.loads realiza después el parseo completo.
def check_json_depth(text):
    # Anidamiento actual; comienza fuera de cualquier objeto o lista.
    depth = 0
    # Estas banderas distinguen estructura JSON de caracteres que forman parte de una cadena.
    quoted = escaped = False
    # Inspecciona carácter a carácter sin interpretar llaves dentro de strings como estructura.
    for char in text:
        # Dentro de una cadena, llaves y corchetes son texto; solo seguimos escapes y cierre de comillas.
        if quoted:
            # El carácter anterior era una barra de escape: este carácter no cierra la cadena.
            if escaped:
                # Consume el escape pendiente y continúa leyendo el contenido de la cadena.
                escaped = False
            # Una barra inversa dentro del texto marca escape del siguiente carácter.
            elif char == "\\":
                # Recuerda que el próximo carácter está escapado.
                escaped = True
            # Una comilla sin escape termina la cadena o abre una nueva según la rama actual.
            elif char == '"':
                # Cierra la cadena actual; las siguientes llaves vuelven a contar como estructura.
                quoted = False
        # Una comilla sin escape termina la cadena o abre una nueva según la rama actual.
        elif char == '"':
            # Abre una cadena de texto; sus llaves no cuentan como estructura.
            quoted = True
        # Fuera de strings, abrir un objeto/lista aumenta el anidamiento.
        elif char in "[{":
            # Incrementa profundidad y comprueba enseguida el máximo admitido.
            depth += 1
            # Validación previa: JSON excede la profundidad permitida.
            if depth > MAX_JSON_DEPTH:
                raise ValueError("JSON excede la profundidad permitida")
        # Cerrar un objeto/lista reduce el anidamiento; el parseo posterior valida su sintaxis.
        elif char in "]}":
            # Sale un nivel de estructura JSON.
            depth -= 1


# Lee bloques HTTP con límites de bytes y tiempo y devuelve un objeto JSON.
# La respuesta debe haberse solicitado con stream=True.
# El finally cierra la conexión tanto en éxito como en error.
def bounded_json(response, max_bytes=MAX_RESPONSE_BYTES, max_seconds=120):
    """El llamador DEBE solicitar stream=True para impedir buffering previo.

    Se cuentan bytes descomprimidos y no se confía en Content-Length.
    El plazo se comprueba entre bloques; el timeout HTTP limita cada lectura.
    """
    try:
        # Content-Length es solo una comprobación previa; puede faltar o describir el tamaño comprimido.
        length = response.headers.get("Content-Length")
        # Validación previa: Respuesta HTTP excede el límite de bytes.
        if length is not None and (int(length) < 0 or int(length) > max_bytes):
            raise ValueError("Respuesta HTTP excede el límite de bytes")
        # Acumula bytes para detectar después la codificación correcta del JSON.
        body = bytearray()
        # Reloj monotónico de lectura: mide el plazo entre bloques recibidos.
        start = time.monotonic()
        # Requests entrega bloques descomprimidos; los límites se comprueban antes de acumularlos.
        for chunk in response.iter_content(chunk_size=8192):
            # Comprueba el plazo entre bloques; no puede interrumpir una lectura bloqueada antes del timeout HTTP.
            if time.monotonic() - start > max_seconds:
                raise ValueError("Respuesta HTTP excede el plazo de lectura")
            # Usa bytes realmente recibidos, no confía en la cabecera declarada por el servidor.
            if len(body) + len(chunk) > max_bytes:
                raise ValueError("Respuesta HTTP excede el límite de bytes")
            # Acumula el bloque solo después de comprobar plazo y bytes reales.
            body.extend(chunk)
        # Detecta BOM/UTF antes de decodificar; permite UTF-8, UTF-16 y UTF-32.
        text = body.decode(json.detect_encoding(body))
        # Comprueba profundidad sobre texto decodificado antes de construir estructuras Python.
        check_json_depth(text)
        # Convierte el cuerpo recibido a valores Python; exige después un objeto raíz.
        data = json.loads(text)
        # Validación previa: Respuesta HTTP: se esperaba un objeto JSON.
        if not isinstance(data, dict):
            raise ValueError("Respuesta HTTP: se esperaba un objeto JSON")
        return data
    finally:
        # Libera la conexión incluso cuando el límite o el parseo causan una excepción.
        response.close()


# Valida que value sea una lista acotada de diccionarios y devuelve esa lista.
def objects(value, limit=MAX_PARTS):
    # Validación previa: Lista de respuesta mal formada o demasiado larga.
    if (
        not isinstance(value, list)
        or len(value) > limit
        or any(not isinstance(x, dict) for x in value)
    ):
        raise ValueError("Lista de respuesta mal formada o demasiado larga")
    return value


# Extrae los IDs de un catálogo compatible y rechaza valores vacíos o mal formados.
def model_ids(data):
    # Valida primero que el catálogo sea una lista limitada de objetos.
    entries = objects(data.get("data", []), MAX_MODELS)
    # Extrae el campo id de cada entrada; los ausentes serán rechazados por la comprobación siguiente.
    ids = [entry.get("id") for entry in entries]
    # Validación previa: Identificador de modelo mal formado o demasiado largo.
    if any(
        not isinstance(value, str) or not value or len(value) > 1024 for value in ids
    ):
        raise ValueError("Identificador de modelo mal formado o demasiado largo")
    return ids


# Acepta texto o None y limita su tamaño de transporte; no lo recorta ni repara.
# La validación docente aplica después su límite menor sin perder el original.
def model_text(value):
    # El transporte ya limita a 1 MiB. Conservar ese texto completo permite
    # marcar como inválido un resultado >16 KiB sin perder la evidencia.
    # Validación previa: Texto del modelo mal formado o demasiado largo.
    if value is not None and (
        not isinstance(value, str) or len(value) > MAX_RESPONSE_BYTES
    ):
        raise ValueError("Texto del modelo mal formado o demasiado largo")
    return value


# Une bloques de texto del proveedor usando la clave indicada y comprueba el tamaño total.
def joined_text(parts, key):
    # Obtiene el texto de cada bloque validado antes de comprobar la suma de tamaños.
    texts = [model_text(part.get(key, "")) for part in objects(parts)]
    # Validación previa: Texto del modelo mal formado o demasiado largo.
    if any(text is None for text in texts) or sum(map(len, texts)) > MAX_RESPONSE_BYTES:
        raise ValueError("Texto del modelo mal formado o demasiado largo")
    return "".join(texts)


# Devuelve la primera alternativa con un message válido de una respuesta compatible.
def first_choice(data):
    # Valida la lista de alternativas; esta práctica utiliza solo la primera.
    choices = objects(data.get("choices", []), 8)
    # Validación previa: Respuesta sin choice/message válido.
    if not choices or not isinstance(choices[0].get("message"), dict):
        raise ValueError("Respuesta sin choice/message válido")
    return choices[0]
