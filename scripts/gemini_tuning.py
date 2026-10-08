"""Preparar/consultar tuning; crear exige proyecto, bucket y presupuesto explícitos."""
import _bootstrap  # noqa: F401
import argparse,json,os,time
from pathlib import Path
from curso.common import ROOT,assert_frozen
from curso.remote import google_client,load_config


def job_spec(project,bucket):
    if not project or not bucket:raise ValueError("Proyecto del curso y bucket son obligatorios")
    if "/" in project:raise ValueError("Usa ID de proyecto, no URL")
    bucket=bucket.removeprefix("gs://").strip("/")
    if "/" in bucket:raise ValueError("Usa nombre de bucket sin subcarpetas")
    return {"project":project,"location":"europe-west4","base_model":"gemini-3.1-flash-lite",
            "train_uri":f"gs://{bucket}/thepower-s7-s8/train.jsonl",
            "validation_uri":f"gs://{bucket}/thepower-s7-s8/validacion.jsonl",
            "epochs":2,"adapter_size":8,"serving_location":"eu"}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("accion",choices=("preparar","subir-datos","crear","estado"))
    p.add_argument("--proyecto");p.add_argument("--bucket");p.add_argument("--job")
    p.add_argument("--presupuesto-usd",type=float);p.add_argument("--coste-estimado-usd",type=float)
    args=p.parse_args();load_config();frozen=assert_frozen()
    project=args.proyecto or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:p.error("Indica el proyecto del curso; no se toma un proyecto personal por defecto")
    os.environ["GOOGLE_CLOUD_PROJECT"]=project
    if args.accion=="estado":
        if not args.job:p.error("--job es obligatorio")
        client=google_client("europe-west4")
        try: print(client.tunings.get(name=args.job).model_dump_json(indent=2))
        finally:client.close()
        return
    spec=job_spec(project,args.bucket or os.getenv("TUNING_BUCKET"))
    if args.accion=="preparar":
        print(json.dumps({"spec":spec,"experimento":frozen,"estado":"PREPARADO; no se ha llamado a Google"},indent=2));return
    if args.presupuesto_usd is None or args.coste_estimado_usd is None:
        p.error("Indica presupuesto autorizado y estimación previa; esto no crea un límite de gasto en Google")
    if not 0 <= args.coste_estimado_usd <= args.presupuesto_usd or args.presupuesto_usd<=0:
        p.error("Estimación fuera del presupuesto")
    if args.accion=="subir-datos":
        from google.cloud import storage
        client=storage.Client(project=project)
        try:
            bucket=client.bucket(spec["train_uri"].split('/')[2])
            for split in ("train","validacion"):
                bucket.blob(f"thepower-s7-s8/{split}.jsonl").upload_from_filename(ROOT/f"datos/gemini/{split}.jsonl",content_type="application/jsonl",if_generation_match=0)
        finally:client.close()
        print("Subidos train y validación; test no se ha subido")
    else:
        from google.genai import types
        client=google_client(spec["location"])
        try:
            job=client.tunings.tune(base_model=spec["base_model"],
                training_dataset=types.TuningDataset(gcs_uri=spec["train_uri"]),
                config=types.CreateTuningJobConfig(tuned_model_display_name="thepower-soporte-s8",
                    epoch_count=spec["epochs"],adapter_size=spec["adapter_size"],
                    validation_dataset=types.TuningDataset(gcs_uri=spec["validation_uri"])))
            path=ROOT/"resultados/google-tuning";path.mkdir(parents=True,exist_ok=True)
            (path/f"job-{time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())}.json").write_text(job.model_dump_json(indent=2))
            print(json.dumps({"job":job.name,"state":str(job.state),"presupuesto_usd":args.presupuesto_usd,
                              "advertencia":"El presupuesto anotado no es un tope técnico del proveedor"}))
        finally:client.close()


if __name__=="__main__":main()
