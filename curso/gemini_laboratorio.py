"""Operaciones del laboratorio público, con los recursos propios del alumno.

Preparar/comprobar/seleccionar/resumir son offline. Subir/crear/consultar/evaluar
requieren autorización explícita del operador. Importar este módulo no llama a
Google ni carga credenciales. Los recibos y originales se guardan en resultados/.
"""

import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, read_jsonl
from .http import bounded_json
from .negocio import CONDITIONS, DATA, GENERATION, dataset_check, grade, payload, sha, summary

BASE = "gemini-3.1-flash-lite"
REGIONS = {"europe-west4": "eu", "us-central1": "us"}
TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def now():
    return datetime.now(timezone.utc).isoformat()


def save_new(path, value):
    """Modo x: conserva un recibo anterior, incluso después de un fallo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} debe ser positivo y finito")
    return value


def prepare(run, project, bucket, prefix, budget, epochs=8, region="europe-west4", rate=3.0, reserve=0.5):
    """Configura un ensayo nuevo sin red y sin tocar ADC ni gcloud global."""
    dataset_check()
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", project or ""):
        raise ValueError("Usa tu ID de proyecto Cloud, no número o URL")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket or ""):
        raise ValueError("Usa tu nombre simple de bucket, sin gs:// ni carpeta")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", prefix or ""):
        raise ValueError("El prefijo/nombre admite letras minúsculas, números y guiones")
    if type(epochs) is not int or not 1 <= epochs <= 20 or region not in REGIONS:
        raise ValueError("En este laboratorio: 1–20 épocas; europe-west4 o us-central1")
    config = {"project": project, "bucket": bucket, "prefix": prefix, "region": region,
              "serving_location": REGIONS[region], "base_model": BASE, "epochs": epochs,
              "budget_usd": positive(budget, "Presupuesto"),
              "training_rate_usd_mtok": positive(rate, "Tarifa"),
              "inference_reserve_usd": positive(reserve, "Reserva"),
              "prepared_at_utc": now(), "generation": GENERATION,
              "dataset_manifest_sha256": sha(DATA / "manifiesto.json"),
              "code_sha256": {name: sha(ROOT / name) for name in
                               ("curso/negocio.py", "curso/gemini_laboratorio.py")},
              "selection_rule": "first, middle, last; max validación completa; empate posterior",
              "note": "Presupuesto estimado local, no un tope técnico de facturación"}
    save_new(Path(run) / "config.json", config)
    return config


def load_run(run):
    dataset_check()
    cfg = read(Path(run) / "config.json")
    if cfg["dataset_manifest_sha256"] != sha(DATA / "manifiesto.json"):
        raise ValueError("Dataset distinto del registrado en este laboratorio")
    for name, expected in cfg["code_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Código cambiado tras preparar: crea otra versión/laboratorio")
    return cfg


def authorized(flag):
    if not flag:
        raise ValueError("Esta acción requiere --autorizar-gasto; revisa destino y presupuesto")


def credentials(cfg):
    """ADC del alumno, cuota de SU proyecto solo en estas peticiones."""
    import google.auth
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return creds.with_quota_project(cfg["project"])


def vertex_client(cfg):
    from google import genai
    from google.genai import types
    return genai.Client(vertexai=True, project=cfg["project"], location=cfg["region"],
                        credentials=credentials(cfg),
                        # Un POST de creación incierto no se repite automáticamente.
                        http_options=types.HttpOptions(api_version="v1", timeout=60000,
                                                       retry_options=types.HttpRetryOptions(attempts=1)))


class Predictor:
    """REST multirregión: cuerpo acotado, sin redirecciones ni corrección de respuestas."""
    def __init__(self, cfg):
        from google.auth.transport.requests import AuthorizedSession
        self.cfg = cfg
        self.session = AuthorizedSession(credentials(cfg), max_refresh_attempts=0)

    def close(self):
        self.session.close()

    def request(self, url, body=None):
        # Solo URLs construidas por este módulo, nunca URLs procedentes del modelo.
        response = (self.session.get(url, timeout=60, stream=True, allow_redirects=False)
                    if body is None else self.session.post(url, json=body, timeout=60,
                                                          stream=True, allow_redirects=False))
        if response.status_code != 200:
            response.close()
            raise RuntimeError(f"Google HTTP {response.status_code}; revisa identidad, recurso, cuota y permisos")
        return bounded_json(response)

    def project_number(self):
        return str(self.request(f"https://cloudresourcemanager.googleapis.com/v1/projects/{self.cfg['project']}")["projectNumber"])

    def base_resource(self):
        return f"projects/{self.cfg['project']}/locations/{self.cfg['serving_location']}/publishers/google/models/{BASE}"

    def generate(self, question, resource, with_catalog=False):
        location = self.cfg["serving_location"]
        url = f"https://aiplatform.{location}.rep.googleapis.com/v1/{resource}:generateContent"
        start = time.perf_counter()
        result = self.request(url, payload(question, with_catalog))
        # El extractor compartido conserva el texto original y excluye pensamiento.
        from .remote import gemini_reply
        reply = gemini_reply(result)
        return {**reply, "latencia_s": time.perf_counter() - start, "model": resource,
                "parametros_solicitados": GENERATION, "con_catalogo": with_catalog}


def estimate(run, cfg):
    """countTokens lee el contador; no genera texto ni crea un entrenamiento."""
    path = Path(run) / "presupuesto.json"
    if path.exists():
        return read(path)
    records = read_jsonl(DATA / "gemini/train.jsonl")
    text = "\n".join(r["systemInstruction"]["parts"][0]["text"] + "\n" +
                     r["contents"][0]["parts"][0]["text"] + "\n" +
                     r["contents"][1]["parts"][0]["text"] for r in records)
    predictor = Predictor(cfg)
    try:
        location = cfg["serving_location"]
        reply = predictor.request(
            f"https://aiplatform.{location}.rep.googleapis.com/v1/{predictor.base_resource()}:countTokens",
            {"contents": [{"role": "user", "parts": [{"text": text}]}]})
    finally:
        predictor.close()
    count = reply["totalTokens"]
    if type(count) is not int or count <= 0:
        raise ValueError("Contador de tokens inválido")
    upper = math.ceil(count * 1.1) + len(records) * 30
    training = upper * cfg["epochs"] * cfg["training_rate_usd_mtok"] / 1e6
    value = {"count_tokens_aggregated": count, "estimated_tokens_per_epoch_with_margin": upper,
             "training_estimated_usd": training,
             "total_estimated_usd": training + cfg["inference_reserve_usd"],
             "config_sha256": sha(Path(run) / "config.json"), "created_at_utc": now(),
             "note": "Estimación con margen, no factura ni garantía de máximo"}
    save_new(path, value)
    return value


def check_budget(run, cfg):
    value = read(Path(run) / "presupuesto.json")
    if value["config_sha256"] != sha(Path(run) / "config.json"):
        raise ValueError("Presupuesto de otra configuración")
    if not math.isfinite(value["total_estimated_usd"]) or value["total_estimated_usd"] > cfg["budget_usd"]:
        raise ValueError("Estimación fuera del presupuesto: no se sube ni crea un job")
    return value


def upload(run, cfg, allow=False):
    """Subida recuperable: solo reutiliza objetos existentes con el MISMO hash."""
    authorized(allow)
    check_budget(run, cfg)
    from google.cloud import storage
    predictor = Predictor(cfg)
    try:
        number = predictor.project_number()
    finally:
        predictor.close()
    client = storage.Client(project=cfg["project"], credentials=credentials(cfg))
    try:
        bucket = client.get_bucket(cfg["bucket"])
        if str(bucket.project_number) != number:
            raise ValueError("El bucket no pertenece al proyecto configurado por el alumno")
        iam = bucket.iam_configuration
        if not iam.uniform_bucket_level_access_enabled or iam.public_access_prevention != "enforced":
            raise ValueError("Se requiere acceso uniforme y prevención pública enforced")
        expected = {f"{cfg['prefix']}/{split}.jsonl" for split in ("train", "validacion")}
        present = {b.name for b in client.list_blobs(bucket, prefix=cfg["prefix"] + "/")}
        if not present <= expected:
            raise ValueError("El prefijo contiene otros objetos; usa uno nuevo para el laboratorio")
        objects = []
        for split in ("train", "validacion"):
            path = DATA / "gemini" / f"{split}.jsonl"
            blob = bucket.blob(f"{cfg['prefix']}/{split}.jsonl")
            if not blob.exists():
                blob.upload_from_filename(path, content_type="application/jsonl", if_generation_match=0)
            if __import__("hashlib").sha256(blob.download_as_bytes()).hexdigest() != sha(path):
                raise ValueError("Objeto existente/subido con contenido distinto: no se sobrescribe")
            blob.reload()
            objects.append({"name": blob.name, "generation": blob.generation, "sha256": sha(path)})
        value = {"project_number": number, "bucket": bucket.name, "objects": objects,
                 "config_sha256": sha(Path(run) / "config.json"), "test_uploaded": False}
        target = Path(run) / "subida.json"
        if target.exists():
            if read(target) != value:
                raise ValueError("Recibo de subida distinto: conserva el anterior")
        else:
            save_new(target, value)
        return value
    finally:
        client.close()


def validate_job(job, cfg, uploaded):
    """Impide vincular un job de otra cuenta, ubicación, base o dataset."""
    expected = f"projects/{uploaded['project_number']}/locations/{cfg['region']}/tuningJobs/"
    if not job["name"].startswith(expected) or not job["name"][len(expected):].isdigit():
        raise ValueError("El job no pertenece al proyecto/ubicación de este laboratorio")
    spec = job["supervised_tuning_spec"]
    if (job["base_model"] != BASE or
        spec["training_dataset_uri"] != f"gs://{cfg['bucket']}/{cfg['prefix']}/train.jsonl" or
        spec["validation_dataset_uri"] != f"gs://{cfg['bucket']}/{cfg['prefix']}/validacion.jsonl" or
        spec["hyper_parameters"]["epoch_count"] != cfg["epochs"]):
        raise ValueError("El job no usa el protocolo de este laboratorio")


def create(run, cfg, allow=False):
    authorized(allow)
    budget = check_budget(run, cfg)
    uploaded = read(Path(run) / "subida.json")
    if uploaded["config_sha256"] != sha(Path(run) / "config.json"):
        raise ValueError("Recibo de subida de otra configuración")
    if (Path(run) / "job-creado.json").exists():
        raise ValueError("Job ya creado; usa estado, no repitas crear")
    from google.genai import types
    config = types.CreateTuningJobConfig(
        tuned_model_display_name=cfg["prefix"], epoch_count=cfg["epochs"],
        export_last_checkpoint_only=False,
        validation_dataset=types.TuningDataset(gcs_uri=f"gs://{cfg['bucket']}/{cfg['prefix']}/validacion.jsonl"))
    # Si el resultado de red es desconocido, el recibo obliga a reconciliarlo.
    save_new(Path(run) / "creacion-solicitada.json",
             {"config": config.model_dump(mode="json", exclude_none=True), "budget": budget,
              "requested_at_utc": now(), "config_sha256": sha(Path(run) / "config.json")})
    client = vertex_client(cfg)
    try:
        job = client.tunings.tune(base_model=BASE,
                                 training_dataset=types.TuningDataset(gcs_uri=f"gs://{cfg['bucket']}/{cfg['prefix']}/train.jsonl"),
                                 config=config)
    finally:
        client.close()
    value = json.loads(job.model_dump_json())
    validate_job(value, cfg, uploaded)
    save_new(Path(run) / "job-creado.json", value)
    return value


def status(run, cfg, recovery_job=None):
    """Lee el job propio; recovery_job permite reconciliar una creación incierta."""
    uploaded = read(Path(run) / "subida.json")
    if recovery_job:
        if (Path(run) / "job-creado.json").exists() or not (Path(run) / "creacion-solicitada.json").exists():
            raise ValueError("Vincular solo recupera una solicitud pendiente sin recibo de job")
        name = recovery_job
    else:
        name = read(Path(run) / "job-creado.json")["name"]
    expected = f"projects/{uploaded['project_number']}/locations/{cfg['region']}/tuningJobs/"
    if not name.startswith(expected) or not name[len(expected):].isdigit():
        raise ValueError("Recurso de job de otro proyecto o ubicación")
    client = vertex_client(cfg)
    try:
        job = client.tunings.get(name=name)
    finally:
        client.close()
    value = json.loads(job.model_dump_json())
    validate_job(value, cfg, uploaded)
    if recovery_job:
        save_new(Path(run) / "job-creado.json", value)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    save_new(Path(run) / "estados" / f"{stamp}.json", value)
    if value["state"] == "JOB_STATE_SUCCEEDED" and not (Path(run) / "estado-final.json").exists():
        save_new(Path(run) / "estado-final.json", value)
    return value


def endpoint_check(resource, cfg, run):
    number = read(Path(run) / "subida.json")["project_number"]
    expected = f"projects/{number}/locations/{cfg['serving_location']}/endpoints/"
    if not resource.startswith(expected) or not resource[len(expected):].isdigit():
        raise ValueError("Endpoint de otra cuenta o ubicación; usa el devuelto por tu job")
    return resource


def candidates(run, cfg):
    """Registra hasta tres checkpoints antes de mirar cualquier test."""
    target = Path(run) / "candidatos.json"
    if target.exists():
        return read(target)
    job = read(Path(run) / "estado-final.json")
    validate_job(job, cfg, read(Path(run) / "subida.json"))
    if job["state"] != "JOB_STATE_SUCCEEDED":
        raise ValueError("Espera a que el entrenamiento termine correctamente")
    points = job["tuned_model"].get("checkpoints") or [
        {"checkpoint_id": "final", "step": 0, "epoch": cfg["epochs"], "endpoint": job["tuned_model"]["endpoint"]}]
    points = sorted(points, key=lambda x: x["step"])
    picked = [points[i] for i in sorted({0, (len(points) - 1) // 2, len(points) - 1})]
    for point in picked:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", str(point["checkpoint_id"])):
            raise ValueError("ID de checkpoint inválido para un archivo local")
        endpoint_check(point["endpoint"], cfg, run)
    value = {"candidates": picked, "registered_at_utc": now(), "job": job["name"]}
    save_new(target, value)
    return value


def result_path(run, condition, split, checkpoint=None):
    name = f"checkpoint-{checkpoint}" if condition == "ajustado_sin_catalogo" and split == "validacion" else condition
    return Path(run) / split / f"{name}.jsonl"


def choose(run, cfg):
    """Selecciona por validación; jamás por aciertos en los tests publicados."""
    if any(list((Path(run) / split).glob("*.jsonl")) for split in ("test", "test-precios")):
        raise ValueError("Test ya consultado: no cambies la selección")
    expected = read_jsonl(DATA / "validacion.jsonl")
    scored = []
    for cp in candidates(run, cfg)["candidates"]:
        path = result_path(run, "ajustado_sin_catalogo", "validacion", cp["checkpoint_id"])
        rows = read_jsonl(path)
        if [r["id"] for r in rows] != [r["id"] for r in expected]:
            raise ValueError("Validación incompleta o con otro checkpoint")
        for row, source in zip(rows, expected):
            if (row["model"] != cp["endpoint"] or row["split"] != "validacion" or
                row["condicion"] != "ajustado_sin_catalogo" or row["pregunta"] != source["pregunta"] or
                row["config_sha256"] != sha(Path(run) / "config.json")):
                raise ValueError("Validación de otro modelo o protocolo")
            if not row.get("error") and row.get("parametros_solicitados") != GENERATION:
                raise ValueError("Generación distinta en validación")
            row["valoracion"] = grade(row.get("respuesta_original"), source)
        scored.append({**cp, "summary": summary(rows), "validation_sha256": sha(path)})
    best = max(scored, key=lambda cp: (cp["summary"]["caso_completo"], cp["step"]))
    value = {"endpoint": best["endpoint"], "checkpoint_id": best["checkpoint_id"],
             "selected_at_utc": now(), "validation": scored,
             "config_sha256": sha(Path(run) / "config.json")}
    save_new(Path(run) / "seleccion.json", value)
    return value


def model_for(run, cfg, condition, split=None, checkpoint=None):
    if condition not in CONDITIONS:
        raise ValueError("Condición desconocida")
    if condition != "ajustado_sin_catalogo":
        return f"projects/{cfg['project']}/locations/{cfg['serving_location']}/publishers/google/models/{BASE}"
    if split == "validacion":
        options = candidates(run, cfg)["candidates"]
        found = next((cp for cp in options if str(cp["checkpoint_id"]) == str(checkpoint)), None)
        if found is None:
            raise ValueError("Indica --checkpoint de tus candidatos registrados")
        return endpoint_check(found["endpoint"], cfg, run)
    selection = read(Path(run) / "seleccion.json")
    if selection["config_sha256"] != sha(Path(run) / "config.json"):
        raise ValueError("Selección de otra configuración")
    return endpoint_check(selection["endpoint"], cfg, run)


def evaluate(run, cfg, condition, split, checkpoint=None, allow=False):
    authorized(allow)
    if split not in ("validacion", "test", "test-precios"):
        raise ValueError("Partición inválida")
    if split != "validacion" and not (Path(run) / "seleccion.json").exists():
        raise ValueError("Selecciona y registra un checkpoint antes de abrir test")
    model = model_for(run, cfg, condition, split, checkpoint)
    target = result_path(run, condition, split, checkpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Reserva la salida ANTES de conectar: una ejecución parcial tampoco se repite.
    with target.open("x", encoding="utf-8") as output:
        predictor = Predictor(cfg)
        try:
            for row in read_jsonl(DATA / f"{split}.jsonl"):
                try:
                    reply = predictor.generate(row["pregunta"], model, condition == "base_con_catalogo_control")
                    error = None
                except Exception as exc:
                    # No registra credenciales ni detalles arbitrarios de una excepción.
                    reply = {"respuesta_original": None, "tokens_entrada": None, "tokens_salida": None}
                    error = {"tipo": type(exc).__name__, "nota": "Petición fallida; conservada en el denominador"}
                item = {"id": row["id"], "pregunta": row["pregunta"], **reply, "model": model,
                        "condicion": condition, "split": split, "fecha_utc": now(), "error": error,
                        "config_sha256": sha(Path(run) / "config.json"),
                        "valoracion": grade(reply.get("respuesta_original"), row)}
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
                output.flush()
                print(row["id"], "OK" if item["valoracion"]["caso_completo"] else "INCOMPLETO", flush=True)
        finally:
            predictor.close()
    return summary(read_jsonl(target))


def query(run, cfg, condition, question, allow=False):
    authorized(allow)
    if not question.strip() or len(question) > 4096:
        raise ValueError("Pregunta vacía o demasiado larga")
    model = model_for(run, cfg, condition)
    predictor = Predictor(cfg)
    try:
        reply = predictor.generate(question, model, condition == "base_con_catalogo_control")
    finally:
        predictor.close()
    value = {"pregunta": question, **reply, "condicion": condition, "fecha_utc": now()}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    save_new(Path(run) / "consultas" / f"{stamp}.json", value)
    return value


def compare(run, cfg, split):
    """Informe offline completo: una condición parcial no se convierte en resultado."""
    if split not in ("test", "test-precios"):
        raise ValueError("Compara test o test-precios")
    expected = read_jsonl(DATA / f"{split}.jsonl")
    selection = read(Path(run) / "seleccion.json")
    result = {}
    hashes = {}
    for condition in CONDITIONS:
        path = result_path(run, condition, split)
        rows = read_jsonl(path)
        if [r["id"] for r in rows] != [r["id"] for r in expected]:
            raise ValueError("Test incompleto o IDs distintos")
        model = model_for(run, cfg, condition)
        for row, source in zip(rows, expected):
            if (row["config_sha256"] != sha(Path(run) / "config.json") or row["model"] != model or
                row["split"] != split or row["condicion"] != condition or row["pregunta"] != source["pregunta"] or
                row["fecha_utc"] < selection["selected_at_utc"]):
                raise ValueError("Resultados de otro protocolo/modelo o anteriores a la selección")
            if not row.get("error") and row.get("parametros_solicitados") != GENERATION:
                raise ValueError("Generación distinta entre condiciones")
            # Recalcular desde original: no confiar en un campo de puntuación editado.
            row["valoracion"] = grade(row.get("respuesta_original"), source)
        result[condition] = summary(rows)
        hashes[condition] = sha(path)
    n = len(expected)
    gain = 100 * (result["ajustado_sin_catalogo"]["caso_completo"] - result["base_sin_catalogo"]["caso_completo"]) / n
    value = {"split": split, "summaries": result, "gain_percentage_points": gain,
             "result_hashes": hashes, "selected_endpoint": selection["endpoint"],
             "note": "Test docente publicado; cobertura automática y revisión de contenido separadas"}
    save_new(Path(run) / split / "comparacion.json", value)
    lines = ["# Comparación de Nébula Hogar", "", "| Condición | Completos | JSON | Esquema | Errores API |",
             "|---|---:|---:|---:|---:|"]
    for condition, score in result.items():
        lines.append(f"| {condition} | {score['caso_completo']}/{n} | {score['json_valido']} | {score['esquema_valido']} | {score['errores_api']} |")
    lines += ["", f"Ventaja: {gain:.2f} puntos porcentuales.", "", value["note"]]
    with (Path(run) / split / "comparacion.md").open("x", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")
    return value
