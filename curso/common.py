"""Datos, prompts y protocolo compartido, sin red."""
import hashlib
import json
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
CATEGORIES = ("acceso", "facturacion", "tecnico", "consulta")
PRIORITIES = ("alta", "media", "baja")
MODEL_ID = os.getenv("OPEN_MODEL_ID", "Qwen/Qwen3-0.6B")
REVISION = os.getenv("OPEN_MODEL_REVISION", "c1899de289a04d12100db370d81485cdf75e47ca")
MAX_NEW_TOKENS = 180
MAX_MODEL_TEXT_CHARS = 16384
GENERATION = {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def basic_prompt():
    return (ROOT / "prompts/basico.txt").read_text().strip()


def fewshot_ids():
    return json.loads((ROOT / "prompts/fewshot-ids.json").read_text())


def messages(incident, fewshot=False):
    result = [{"role": "system", "content": basic_prompt()}]
    if fewshot:
        by_id = {row["id"]: row for row in read_jsonl(ROOT / "datos/train.jsonl")}
        for row_id in fewshot_ids():
            row = by_id[row_id]
            result.extend([
                {"role": "user", "content": row["incidencia"]},
                {"role": "assistant", "content": json.dumps(row["objetivo"], ensure_ascii=False)},
            ])
    result.append({"role": "user", "content": incident})
    return result


def validate_answer(raw, expected):
    """JSON estricto: no extrae fences, no repara ni reintenta."""
    out = {"json_valido": False, "categoria_correcta": False,
           "prioridad_correcta": False, "ambas_correctas": False,
           "esquema_valido": False, "motivo": None}
    if not isinstance(raw, str) or len(raw) > MAX_MODEL_TEXT_CHARS:
        out["motivo"] = "Texto no válido o demasiado largo"
        return out
    try:
        from .http import check_json_depth
        check_json_depth(raw)
        obj = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, TypeError, RecursionError) as exc:
        out["motivo"] = type(exc).__name__
        return out
    out["json_valido"] = True
    if not isinstance(obj, dict):
        out["motivo"] = "El JSON no es un objeto"
        return out
    out["esquema_valido"] = (set(obj) == {"categoria", "prioridad", "respuesta"}
                            and obj.get("categoria") in CATEGORIES
                            and obj.get("prioridad") in PRIORITIES
                            and isinstance(obj.get("respuesta"), str)
                            and bool(obj["respuesta"].strip()))
    # Un objeto interpretable permite medir etiquetas aunque falle otro campo.
    out["categoria_correcta"] = obj.get("categoria") == expected["categoria"]
    out["prioridad_correcta"] = obj.get("prioridad") == expected["prioridad"]
    out["ambas_correctas"] = out["categoria_correcta"] and out["prioridad_correcta"]
    return out


def validate_splits(folder=None):
    folder = Path(folder or ROOT / "datos")
    splits = {name: read_jsonl(folder / f"{name}.jsonl") for name in ("train", "validacion", "test")}
    ids, texts, groups = set(), set(), set()
    for name, rows in splits.items():
        local_groups = set()
        for row in rows:
            if row["id"] in ids:
                raise ValueError("ID repetido")
            normalized = " ".join(row["incidencia"].casefold().split())
            if normalized in texts:
                raise ValueError("Incidencia duplicada")
            if row["familia"] in groups:
                raise ValueError("Familia de paráfrasis mezclada entre splits")
            ids.add(row["id"]); texts.add(normalized); local_groups.add(row["familia"])
            if not validate_answer(json.dumps(row["objetivo"]), row["objetivo"])["esquema_valido"]:
                raise ValueError("Objetivo inválido")
        groups |= local_groups
    counts = {name: len(rows) for name, rows in splits.items()}
    if not (150 <= counts["train"] <= 300 and counts["validacion"] == counts["test"] == 30):
        raise ValueError(f"Particiones incorrectas: {counts}")
    return counts


def experiment_manifest():
    files = [ROOT / f"datos/{s}.jsonl" for s in ("train", "validacion", "test")]
    files += [ROOT / "prompts/basico.txt", ROOT / "prompts/fewshot-ids.json"]
    return {"model": MODEL_ID, "revision": REVISION, "generation": GENERATION,
            "enable_thinking": False, "files": {str(p.relative_to(ROOT)): digest(p) for p in files}}


def assert_frozen():
    expected = json.loads((ROOT / "config/experimento-congelado.json").read_text())
    if expected != experiment_manifest():
        raise ValueError("Datos o protocolo cambiaron después de congelar el experimento")
    return expected
