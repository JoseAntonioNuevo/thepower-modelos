# GUÍA DE LECTURA · curso/common.py
# Punto común del experimento: datos, instrucciones, modelo y evaluación.
# Lee JSONL y prepara mensajes; valida respuestas sin arreglarlas.
# El manifiesto conserva hashes para detectar cambios en datos o configuración.
# Una consulta utiliza incidencia y prompt; la respuesta objetivo solo enseña o evalúa.
# None en consumo significa desconocido, no cero.

"""Datos, prompts y protocolo compartido, sin red."""

# Calcula hashes SHA-256 para identificar el contenido de datos y adaptadores.
import hashlib

# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
import json

# Lee configuración del entorno y ajusta opciones del proceso.
import os

# Path permite construir rutas, comprobar archivos y leer/escribir sin concatenar separadores a mano.
from pathlib import Path

# Carga variables desde .env sin incluir claves en el código.
from dotenv import load_dotenv

# Localiza la raíz desde este archivo para que las rutas no dependan del directorio de la terminal.
ROOT = Path(__file__).resolve().parents[1]
# Carga opciones locales sin reemplazar variables ya definidas; nunca se imprimen sus valores aquí.
load_dotenv(ROOT / ".env", override=False)
# Taxonomía permitida del ejercicio; el validador rechaza valores ajenos a esta lista.
CATEGORIES = ("acceso", "facturacion", "tecnico", "consulta")
# Tres niveles de prioridad definidos por el prompt docente.
PRIORITIES = ("alta", "media", "baja")
# Selecciona el modelo configurado o Qwen3-0.6B como valor del curso.
MODEL_ID = os.getenv("OPEN_MODEL_ID", "Qwen/Qwen3-0.6B")
# Fija el commit de los pesos/tokenizer para comparar la misma versión.
REVISION = os.getenv("OPEN_MODEL_REVISION", "c1899de289a04d12100db370d81485cdf75e47ca")
# Límite de salida por consulta; no es la longitud máxima del contexto de entrenamiento.
MAX_NEW_TOKENS = 180
# Límite docente de texto evaluable; una salida mayor se conserva pero no pasa el validador.
MAX_MODEL_TEXT_CHARS = 16384
# Greedy (sin muestreo) y máximo común de salida para original, few-shot y LoRA.
GENERATION = {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS}


# Recibe una ruta y devuelve su SHA-256: identifica el contenido exacto del archivo.
def digest(path):
    # Devuelve el hash hexadecimal; cambiar cualquier byte del archivo cambia su identidad.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Lee un archivo JSONL y devuelve una lista de diccionarios.
# Cada línea no vacía representa un ejemplo independiente.
def read_jsonl(path):
    # Parsea cada línea no vacía y reúne todos los ejemplos; un JSON mal formado produce excepción.
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


# Escribe registros como un JSON por línea y crea la carpeta de destino si falta.
# La ruta la decide el llamador; esta función sí escribe en disco.
def write_jsonl(path, rows):
    # Ruta concreta de lectura/escritura; el código utiliza Path para gestionar carpetas y archivos.
    path = Path(path)
    # Crea la carpeta necesaria; parents/exist_ok controlan subcarpetas y rutas ya existentes.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Escribe el contenido legible en disco para conservar configuración o resultados de este paso.
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


# Devuelve las instrucciones comunes como texto, quitando espacios externos.
def basic_prompt():
    # Lee las mismas instrucciones que utilizarán entrenamiento, original y LoRA en inferencia.
    return (ROOT / "prompts/basico.txt").read_text().strip()


# Devuelve los IDs de los cuatro ejemplos elegidos de train antes de evaluar test.
def fewshot_ids():
    # Lee la selección fijada por ID, sin elegir ejemplos según el resultado del test.
    return json.loads((ROOT / "prompts/fewshot-ids.json").read_text())


# Construye mensajes system/user/assistant para una consulta.
# Con fewshot=True intercala ejemplos de train y al final añade la incidencia nueva.
def messages(incident, fewshot=False):
    # Comienza la conversación con una única instrucción system compartida por todas las condiciones.
    result = [{"role": "system", "content": basic_prompt()}]
    # Añade ejemplos como mensajes del contexto; no modifica ningún peso.
    if fewshot:
        # Indexa solo train por ID para evitar tomar ejemplos de validación o test.
        by_id = {row["id"]: row for row in read_jsonl(ROOT / "datos/train.jsonl")}
        # Recorre los cuatro IDs fijados; cada uno debe existir en el diccionario de train.
        for row_id in fewshot_ids():
            # Un ejemplo: ID/metadatos para nosotros e incidencia/objetivo para la tarea.
            row = by_id[row_id]
            # Por cada ejemplo agrega una pareja user/assistant, mostrando la incidencia y su respuesta objetivo.
            result.extend(
                [
                    {"role": "user", "content": row["incidencia"]},
                    {
                        "role": "assistant",
                        "content": json.dumps(row["objetivo"], ensure_ascii=False),
                    },
                ]
            )
    # La última entrada es siempre la nueva incidencia que queremos clasificar.
    result.append({"role": "user", "content": incident})
    # Entrega la lista de mensajes al cliente/tokenizer; todavía no ha consultado ningún modelo.
    return result


# Recibe texto original y objetivo; devuelve banderas de formato y acierto.
# Comprueba JSON, esquema, categoría y prioridad por separado, sin corregir la salida.
# Una etiqueta puede medirse aunque otro campo invalide el esquema.
def validate_answer(raw, expected):
    """JSON estricto: no extrae fences, no repara ni reintenta."""
    # Todas las banderas empiezan en False; solo pasan a True tras comprobar la propiedad correspondiente.
    out = {
        "json_valido": False,
        "categoria_correcta": False,
        "prioridad_correcta": False,
        "ambas_correctas": False,
        "esquema_valido": False,
        "motivo": None,
    }
    # Descarta entradas ausentes/no textuales y salidas demasiado largas antes del parseo.
    if not isinstance(raw, str) or len(raw) > MAX_MODEL_TEXT_CHARS:
        # Conserva el motivo del rechazo para entender el fallo sin alterar la respuesta original.
        out["motivo"] = "Texto no válido o demasiado largo"
        # Devuelve las banderas acumuladas: los fallos permanecen False para contar en la evaluación.
        return out
    try:
        from .http import check_json_depth

        # Limita anidamiento antes de json.loads para evitar estructuras excesivamente profundas.
        check_json_depth(raw)
        # Parsea el texto entero y rechaza NaN/Infinity; no extrae JSON de Markdown ni corrige texto.
        obj = json.loads(
            raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))
        )
    except (ValueError, TypeError, RecursionError) as exc:
        # Conserva el motivo del rechazo para entender el fallo sin alterar la respuesta original.
        out["motivo"] = type(exc).__name__
        # Devuelve las banderas acumuladas: los fallos permanecen False para contar en la evaluación.
        return out
    # El parseo terminó correctamente; aún falta comprobar objeto, claves y valores permitidos.
    out["json_valido"] = True
    # Un JSON puede ser una lista o un número; esta tarea exige un objeto con campos.
    if not isinstance(obj, dict):
        # Conserva el motivo del rechazo para entender el fallo sin alterar la respuesta original.
        out["motivo"] = "El JSON no es un objeto"
        # Devuelve las banderas acumuladas: los fallos permanecen False para contar en la evaluación.
        return out
    # Exige exactamente tres claves, valores de la taxonomía y una respuesta textual no vacía.
    out["esquema_valido"] = (
        set(obj) == {"categoria", "prioridad", "respuesta"}
        and obj.get("categoria") in CATEGORIES
        and obj.get("prioridad") in PRIORITIES
        and isinstance(obj.get("respuesta"), str)
        and bool(obj["respuesta"].strip())
    )
    # Un objeto interpretable permite medir etiquetas aunque falle otro campo.
    # Compara la etiqueta generada con la anotación del caso, sin comparar marcas de modelos.
    out["categoria_correcta"] = obj.get("categoria") == expected["categoria"]
    # Compara el nivel generado con el esperado para nuestras reglas.
    out["prioridad_correcta"] = obj.get("prioridad") == expected["prioridad"]
    # El acierto conjunto necesita acertar categoría y prioridad en la misma respuesta.
    out["ambas_correctas"] = out["categoria_correcta"] and out["prioridad_correcta"]
    # Devuelve las banderas acumuladas: los fallos permanecen False para contar en la evaluación.
    return out


# Comprueba cantidades, IDs, textos, familias y forma de las respuestas objetivo.
# Una familia puede repetirse dentro de train, pero no cruzar a otra partición.
def validate_splits(folder=None):
    # Carpeta que se recorre para leer lotes o generar una salida temporal.
    folder = Path(folder or ROOT / "datos")
    # Lee las tres particiones como grupos separados antes de comprobar solapamientos.
    splits = {
        name: read_jsonl(folder / f"{name}.jsonl")
        for name in ("train", "validacion", "test")
    }
    # Acumula IDs/textos globales y familias de particiones ya revisadas.
    ids, texts, groups = set(), set(), set()
    # Valida una partición cada vez y contrasta sus familias con las anteriores.
    for name, rows in splits.items():
        # Familias presentes solo en la partición actual; se agregan al conjunto global al acabar.
        local_groups = set()
        # Recorre todos los ejemplos del lote; un error no elimina el caso de la comparación.
        for row in rows:
            # Validación previa: ID repetido.
            if row["id"] in ids:
                raise ValueError("ID repetido")
            # Ignora mayúsculas y espacios repetidos al detectar incidencias duplicadas.
            normalized = " ".join(row["incidencia"].casefold().split())
            # Validación previa: Incidencia duplicada.
            if normalized in texts:
                raise ValueError("Incidencia duplicada")
            # Validación previa: Familia de paráfrasis mezclada entre splits.
            if row["familia"] in groups:
                raise ValueError("Familia de paráfrasis mezclada entre splits")
            # Registra ID, texto normalizado y familia para detectar duplicados en los siguientes registros.
            ids.add(row["id"])
            texts.add(normalized)
            local_groups.add(row["familia"])
            # Validación previa: Objetivo inválido.
            if not validate_answer(json.dumps(row["objetivo"]), row["objetivo"])[
                "esquema_valido"
            ]:
                raise ValueError("Objetivo inválido")
        # Añade familias al terminar la partición: repetición dentro de train se permite; entre particiones se rechaza.
        groups |= local_groups
    # Cuenta registros por partición; cantidad de filas no equivale a diversidad de situaciones.
    counts = {name: len(rows) for name, rows in splits.items()}
    # Exige el rango comunicado de train y exactamente 30 registros en validación y test.
    if not (
        150 <= counts["train"] <= 300 and counts["validacion"] == counts["test"] == 30
    ):
        raise ValueError(f"Particiones incorrectas: {counts}")
    # Devuelve las cantidades solo si las particiones cumplen todas las comprobaciones.
    return counts


# Describe modelo, revisión, generación y hashes de datos/prompts.
# Este diccionario permite comprobar si dos ejecuciones usan el mismo protocolo.
def experiment_manifest():
    # Datos de las tres particiones cuyos contenidos quedan identificados por hash.
    files = [ROOT / f"datos/{s}.jsonl" for s in ("train", "validacion", "test")]
    # También fija las instrucciones y la selección few-shot para no cambiar la comparación.
    files += [ROOT / "prompts/basico.txt", ROOT / "prompts/fewshot-ids.json"]
    # Conserva revisión, generación y hashes como descripción reproducible del experimento.
    return {
        "model": MODEL_ID,
        "revision": REVISION,
        "generation": GENERATION,
        "enable_thinking": False,
        "files": {str(p.relative_to(ROOT)): digest(p) for p in files},
    }


# Compara el manifiesto guardado con el calculado ahora; devuelve el original.
# Si cambió algún contenido o ajuste, detiene la ejecución antes de continuar.
def assert_frozen():
    # Lee el manifiesto anterior congelado; sus hashes no se actualizan al validar.
    expected = json.loads((ROOT / "config/experimento-congelado.json").read_text())
    # Validación previa: Datos o protocolo cambiaron después de congelar el experimento.
    if expected != experiment_manifest():
        raise ValueError(
            "Datos o protocolo cambiaron después de congelar el experimento"
        )
    # Entrega el manifiesto verificado para incluirlo en cada resultado.
    return expected
