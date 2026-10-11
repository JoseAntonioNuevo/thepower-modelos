# GUÍA DE LECTURA · scripts/gemini_tuning.py
# Cuatro acciones distintas: preparar, subir datos, crear tuning y consultar estado.
# Preparar solo describe el trabajo; las otras acciones acceden a Google Cloud.
# Train y validación se suben; test queda fuera del trabajo de entrenamiento.
# Proyecto, bucket y presupuesto deben ser explícitos.
# El presupuesto anotado es una comprobación local, no un límite de gasto del proveedor.

"""Preparar/consultar tuning; crear exige proyecto, bucket y presupuesto explícitos."""

# Hace visible la raíz del proyecto al ejecutar un archivo desde scripts/.
import _bootstrap  # noqa: F401

# Permite leer opciones de la terminal y generar ayuda del comando.
# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
# Lee configuración del entorno y ajusta opciones del proceso.
# Mide duración y prepara fechas; perf_counter/monotonic sirven para intervalos, no fechas del calendario.
import argparse, json, os, time

# Path permite construir rutas, comprobar archivos y leer/escribir sin concatenar separadores a mano.
from pathlib import Path

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.common import ROOT, assert_frozen

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.remote import google_client, load_config


# Valida proyecto y bucket y devuelve la especificación offline de tuning.
# No consulta Google, no crea recursos y nunca incluye el archivo de test.
def job_spec(project, bucket):
    # Validación previa: Proyecto del curso y bucket son obligatorios.
    if not project or not bucket:
        raise ValueError("Proyecto del curso y bucket son obligatorios")
    # Validación previa: Usa ID de proyecto, no URL.
    if "/" in project:
        raise ValueError("Usa ID de proyecto, no URL")
    # Acepta nombre simple o gs://nombre, retirando únicamente prefijo y barras externas.
    bucket = bucket.removeprefix("gs://").strip("/")
    # Validación previa: Usa nombre de bucket sin subcarpetas.
    if "/" in bucket:
        raise ValueError("Usa nombre de bucket sin subcarpetas")
    return {
        "project": project,
        "location": "europe-west4",
        "base_model": "gemini-3.1-flash-lite",
        "train_uri": f"gs://{bucket}/thepower-s7-s8/train.jsonl",
        "validation_uri": f"gs://{bucket}/thepower-s7-s8/validacion.jsonl",
        "epochs": 2,
        "adapter_size": 8,
        "serving_location": "eu",
    }


# Despacha la acción CLI: la preparación es offline; subir/crear/estado usan Google.
def main():
    # Parser de argumentos: describe las opciones que admite este comando.
    p = argparse.ArgumentParser()
    # Opción de terminal 'accion': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    p.add_argument("accion", choices=("preparar", "subir-datos", "crear", "estado"))
    # Opción de terminal '--proyecto': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    # Opción de terminal '--bucket': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    # Opción de terminal '--job': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    p.add_argument("--proyecto")
    p.add_argument("--bucket")
    p.add_argument("--job")
    # Opción de terminal '--presupuesto-usd': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    # Opción de terminal '--coste-estimado-usd': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    p.add_argument("--presupuesto-usd", type=float)
    p.add_argument("--coste-estimado-usd", type=float)
    # Argumentos de la terminal convertidos a atributos como args.salida o args.modelo.
    # Carga configuración del entorno antes de comprobar opciones o crear una petición.
    # Manifiesto verificado que acompaña los resultados y conserva el protocolo exacto.
    args = p.parse_args()
    load_config()
    frozen = assert_frozen()
    # Proyecto explícito del laboratorio que recibirá la llamada Google.
    project = args.proyecto or os.getenv("GOOGLE_CLOUD_PROJECT")
    # Exige proyecto explícito antes de consultar/subir/crear; evita usar por accidente otro proyecto ADC.
    if not project:
        p.error(
            "Indica el proyecto del curso; no se toma un proyecto personal por defecto"
        )
    # Hace que las funciones compartidas utilicen el proyecto ya validado en esta ejecución.
    os.environ["GOOGLE_CLOUD_PROJECT"] = project
    # Despacha una acción específica; preparar/estado/subir/crear tienen efectos diferentes.
    if args.accion == "estado":
        # Para consultar estado exige el ID del trabajo existente; no crea otro entrenamiento.
        if not args.job:
            p.error("--job es obligatorio")
        # Cliente del proveedor creado únicamente para la acción solicitada.
        client = google_client("europe-west4")
        # Muestra el resultado/estado de este paso en la terminal; las claves no se añaden al informe.
        try:
            print(client.tunings.get(name=args.job).model_dump_json(indent=2))
        # Cierra el recurso abierto; libera socket/conexión aunque el paso anterior haya fallado.
        finally:
            client.close()
        return
    # Especificación del trabajo de tuning; describirla no equivale a ejecutarlo.
    spec = job_spec(project, args.bucket or os.getenv("TUNING_BUCKET"))
    # Despacha una acción específica; preparar/estado/subir/crear tienen efectos diferentes.
    if args.accion == "preparar":
        # Muestra el resultado/estado de este paso en la terminal; las claves no se añaden al informe.
        print(
            json.dumps(
                {
                    "spec": spec,
                    "experimento": frozen,
                    "estado": "PREPARADO; no se ha llamado a Google",
                },
                indent=2,
            )
        )
        return
    # Exige estimación y presupuesto como argumentos antes de una acción que consume/escribe recursos.
    if args.presupuesto_usd is None or args.coste_estimado_usd is None:
        # El parser detiene el comando e informa del argumento obligatorio o inválido.
        p.error(
            "Indica presupuesto autorizado y estimación previa; esto no crea un límite de gasto en Google"
        )
    # Rechaza una estimación superior al presupuesto o un presupuesto no positivo.
    if (
        not 0 <= args.coste_estimado_usd <= args.presupuesto_usd
        or args.presupuesto_usd <= 0
    ):
        # El parser detiene el comando e informa del argumento obligatorio o inválido.
        p.error("Estimación fuera del presupuesto")
    # Despacha una acción específica; preparar/estado/subir/crear tienen efectos diferentes.
    if args.accion == "subir-datos":
        from google.cloud import storage

        # Cliente del proveedor creado únicamente para la acción solicitada.
        client = storage.Client(project=project)
        try:
            # Obtiene una referencia al bucket indicado; no crea un bucket nuevo.
            bucket = client.bucket(spec["train_uri"].split("/")[2])
            # Solo sube train y validación; el test se conserva aparte para la evaluación posterior.
            for split in ("train", "validacion"):
                # Escribe el JSONL en Cloud Storage; if_generation_match=0 rechaza sobrescribir un objeto previo.
                bucket.blob(f"thepower-s7-s8/{split}.jsonl").upload_from_filename(
                    ROOT / f"datos/gemini/{split}.jsonl",
                    content_type="application/jsonl",
                    if_generation_match=0,
                )
        # Cierra el recurso abierto; libera socket/conexión aunque el paso anterior haya fallado.
        finally:
            client.close()
        # Muestra el resultado/estado de este paso en la terminal; las claves no se añaden al informe.
        print("Subidos train y validación; test no se ha subido")
    else:
        from google.genai import types

        # Cliente del proveedor creado únicamente para la acción solicitada.
        client = google_client(spec["location"])
        try:
            # Trabajo de tuning creado en Google; su identificador permite consultar el mismo entrenamiento.
            job = client.tunings.tune(
                base_model=spec["base_model"],
                training_dataset=types.TuningDataset(gcs_uri=spec["train_uri"]),
                config=types.CreateTuningJobConfig(
                    tuned_model_display_name="thepower-soporte-s8",
                    epoch_count=spec["epochs"],
                    adapter_size=spec["adapter_size"],
                    validation_dataset=types.TuningDataset(
                        gcs_uri=spec["validation_uri"]
                    ),
                ),
            )
            # Ruta concreta de lectura/escritura; el código utiliza Path para gestionar carpetas y archivos.
            # Crea la carpeta necesaria; parents/exist_ok controlan subcarpetas y rutas ya existentes.
            path = ROOT / "resultados/google-tuning"
            path.mkdir(parents=True, exist_ok=True)
            # Escribe el contenido legible en disco para conservar configuración o resultados de este paso.
            (
                path / f"job-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
            ).write_text(job.model_dump_json(indent=2))
            # Muestra el resultado/estado de este paso en la terminal; las claves no se añaden al informe.
            print(
                json.dumps(
                    {
                        "job": job.name,
                        "state": str(job.state),
                        "presupuesto_usd": args.presupuesto_usd,
                        "advertencia": "El presupuesto anotado no es un tope técnico del proveedor",
                    }
                )
            )
        # Cierra el recurso abierto; libera socket/conexión aunque el paso anterior haya fallado.
        finally:
            client.close()


# Punto de entrada: ejecutar como script llama main; importar el módulo solo define sus funciones.
# Invoca el punto de entrada únicamente cuando se ejecuta este archivo como programa.
if __name__ == "__main__":
    main()
