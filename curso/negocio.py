"""Dataset público de Nébula Hogar y evaluación de cobertura, sin red.

La fuente y los objetivos son sintéticos. El evaluador conserva el texto original:
no extrae Markdown ni repara JSON. Cobertura no equivale a veracidad completa;
las contradicciones y los añadidos falsos requieren revisión contra catálogo.
"""

import hashlib
import json
import re
import unicodedata
from decimal import Decimal
from pathlib import Path

from .common import ROOT, assert_frozen, read_jsonl

DATA = ROOT / "datos/nebula-hogar"
STATES = {"respuesta", "aclaracion", "no_catalogo", "sin_datos", "fuera_ambito"}
GENERATION = {"temperature": 0, "maxOutputTokens": 384,
              "thinkingConfig": {"thinkingLevel": "MINIMAL"}}
CONDITIONS = ("base_sin_catalogo", "ajustado_sin_catalogo", "base_con_catalogo_control")


def sha(path):
    """Identifica los bytes de un archivo; un hash no certifica su calidad."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_check():
    """Comprueba el dataset público y preserva el experimento original de S7."""
    assert_frozen()
    manifest = json.loads((DATA / "manifiesto.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files_sha256"].items():
        if sha(DATA / relative) != expected:
            raise ValueError(f"Dataset modificado: {relative}; crea otra versión")
    catalog = json.loads((DATA / "catalogo.json").read_text(encoding="utf-8"))
    ids = {p["id"] for p in catalog["productos"]}
    for split, count in manifest["splits"].items():
        rows = read_jsonl(DATA / f"{split}.jsonl")
        if len(rows) != count or len({r["id"] for r in rows}) != count:
            raise ValueError(f"Recuento/IDs incorrectos en {split}")
        for row in rows:
            if not grade(json.dumps(row["objetivo"], ensure_ascii=False), row, ids)["caso_completo"]:
                raise ValueError(f"Objetivo que no pasa el contrato: {row['id']}")
    # La conversión contiene solo instrucción, pregunta y respuesta objetivo.
    prompt = (DATA / "prompt.txt").read_text(encoding="utf-8").strip()
    for split in ("train", "validacion"):
        local = read_jsonl(DATA / f"{split}.jsonl")
        converted = read_jsonl(DATA / "gemini" / f"{split}.jsonl")
        if len(local) != len(converted):
            raise ValueError("Conversión incompleta")
        for row, record in zip(local, converted):
            expected = {"systemInstruction": {"parts": [{"text": prompt}]},
                        "contents": [{"role": "user", "parts": [{"text": row["pregunta"]}]},
                                     {"role": "model", "parts": [{"text": json.dumps(row["objetivo"], ensure_ascii=False)}]}]}
            if record != expected:
                raise ValueError(f"Conversión distinta del objetivo: {row['id']}")
    return manifest


def normalized(text):
    """Permite acentos y algunas representaciones de números equivalentes."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    text = text.replace("vatios", "w").replace("mililitros", "ml")
    return re.sub(r"\s+", " ", re.sub(r"[^\w.]+", " ", text)).strip()


def mentions(text, term):
    """45 y 45,00 son equivalentes; 9 no coincide con 19, 90 o 9,90."""
    if term == "?":
        return "?" in text
    value = normalized(term)
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        numbers = re.findall(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w.])", text)
        return any(Decimal(n.replace(",", ".")) == Decimal(value) for n in numbers)
    if value in ("aceptacion", "aceptada", "aceptado"):
        if re.search(r"\bacept(?:acion|ada|ado|a)\b", normalized(text)):
            return True
    variants = {value}
    for word, digit in (("tres", "3"), ("cinco", "5"), ("dos", "2"), ("seis", "6"), ("doce", "12")):
        if word in value:
            variants.add(value.replace(word, digit))
    return any(re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", normalized(text)) for v in variants)


def grade(raw, row, product_ids=None):
    """Mide formato, estado, SKU y cobertura; no hace una revisión semántica."""
    if product_ids is None:
        catalog = json.loads((DATA / "catalogo.json").read_text(encoding="utf-8"))
        product_ids = {p["id"] for p in catalog["productos"]}
    result = {k: False for k in ("json_valido", "esquema_valido", "estado_correcto",
                                "productos_correctos", "hechos_requeridos", "caso_completo")}
    result.update(grupos_ausentes=[], motivo=None)
    if not isinstance(raw, str) or len(raw) > 16384:
        result["motivo"] = "Sin texto o demasiado largo"
        return result
    try:
        obj = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, TypeError, RecursionError):
        result["motivo"] = "JSON inválido"
        return result
    result["json_valido"] = True
    if not isinstance(obj, dict) or set(obj) != {"estado", "productos", "respuesta"}:
        result["motivo"] = "Campos incorrectos"
        return result
    ids, text, state = obj["productos"], obj["respuesta"], obj["estado"]
    schema = (isinstance(state, str) and state in STATES and isinstance(ids, list)
              and all(isinstance(x, str) and x in product_ids for x in ids)
              and len(ids) == len(set(ids)) and isinstance(text, str)
              and 0 < len(text.split()) <= row["criterios"]["max_palabras"])
    result["esquema_valido"] = schema
    result["estado_correcto"] = state == row["objetivo"]["estado"]
    result["productos_correctos"] = (isinstance(ids, list) and all(isinstance(x, str) for x in ids)
                                     and sorted(ids) == sorted(row["objetivo"]["productos"]))
    if isinstance(text, str):
        result["grupos_ausentes"] = [g for g in row["criterios"]["requiere"]
                                     if not any(mentions(text, t) for t in g)]
        result["hechos_requeridos"] = not result["grupos_ausentes"]
    result["caso_completo"] = all(result[k] for k in ("esquema_valido", "estado_correcto",
                                                    "productos_correctos", "hechos_requeridos"))
    return result


def payload(question, with_catalog=False):
    """Inferencia: no incluye la respuesta objetivo ni ejemplos few-shot."""
    system = (DATA / "prompt.txt").read_text(encoding="utf-8").strip()
    if with_catalog:
        system += "\nCONTEXTO AUTORIZADO DEL NEGOCIO:\n" + (DATA / "catalogo.json").read_text(encoding="utf-8")
    return {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": question}]}],
            "generationConfig": GENERATION}


def summary(rows):
    """Los errores permanecen en el denominador y consumo desconocido es null."""
    def total(key):
        values = [r.get(key) for r in rows]
        return sum(values) if values and all(type(v) is int for v in values) else None
    return {"casos": len(rows),
            **{k: sum(r["valoracion"][k] for r in rows)
               for k in ("caso_completo", "json_valido", "esquema_valido", "estado_correcto",
                         "productos_correctos", "hechos_requeridos")},
            "errores_api": sum(bool(r.get("error")) for r in rows),
            "tokens_entrada": total("tokens_entrada"), "tokens_salida": total("tokens_salida")}
