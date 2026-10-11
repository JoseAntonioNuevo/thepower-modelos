# GUÍA DE LECTURA · scripts/demo.py
# Ejecuta una consulta de desarrollo y conserva la respuesta tal como llegó.
# --proveedor elige el recorrido local/servicio; --adaptador activa LoRA con Qwen.
# --fewshot cambia los mensajes enviados, no entrena el modelo.
# La valoración compara con el objetivo y no modifica la respuesta generada.
# --salida guarda evidencia y rechaza sobrescribir un archivo existente.

# Hace visible la raíz del proyecto al ejecutar un archivo desde scripts/.
import _bootstrap  # noqa: F401

# Permite leer opciones de la terminal y generar ayuda del comando.
# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
# Mide duración y prepara fechas; perf_counter/monotonic sirven para intervalos, no fechas del calendario.
import argparse, json, time

# Path permite construir rutas, comprobar archivos y leer/escribir sin concatenar separadores a mano.
from pathlib import Path

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.common import ROOT, read_jsonl, validate_answer, assert_frozen


# Lee el caso y los argumentos, llama al proveedor y registra respuesta/valoración.
def main():
    # Parser de argumentos: describe las opciones que admite este comando.
    p = argparse.ArgumentParser(description="Un caso de desarrollo; no usa el test.")
    # Opción de terminal '--proveedor': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    p.add_argument(
        "--proveedor", choices=("local", "openrouter", "qwen", "google"), required=True
    )
    # Opción de terminal '--modelo': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    # Opción de terminal '--caso': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    p.add_argument("--modelo")
    p.add_argument("--caso", default="DE-01")
    # Opción de terminal '--local-api': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    p.add_argument("--local-api", choices=("native", "openai"), default="native")
    # Opción de terminal '--fewshot': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    # Opción de terminal '--adaptador': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    p.add_argument("--fewshot", action="store_true")
    p.add_argument("--adaptador")
    # Opción de terminal '--salida': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    p.add_argument("--salida")
    # Argumentos de la terminal convertidos a atributos como args.salida o args.modelo.
    args = p.parse_args()
    # Un ejemplo: ID/metadatos para nosotros e incidencia/objetivo para la tarea.
    row = next(
        x for x in read_jsonl(ROOT / "datos/desarrollo.jsonl") if x["id"] == args.caso
    )
    # Ruta Transformers: conserva protocolo y permite original, few-shot o adaptador LoRA.
    if args.proveedor == "qwen":
        # Comprueba hashes del experimento antes de consultar la condición Qwen.
        assert_frozen()
        # Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
        from curso.qwen import load_qwen, generate

        # Carga la misma base y solo aplica adaptador cuando --adaptador contiene una ruta.
        model, tok, device = load_qwen(args.adaptador)
        # Consulta una incidencia de desarrollo, no los casos reservados de test.
        result = generate(model, tok, device, row["incidencia"], args.fewshot)
    else:
        # Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
        from curso.remote import chat_local, chat_openrouter, chat_google

        # Ruta LM Studio: --local-api decide petición nativa o compatible.
        if args.proveedor == "local":
            result = chat_local(
                row["incidencia"], args.modelo, args.fewshot, args.local_api == "native"
            )
        # Ruta del servicio OpenRouter: requiere identificar explícitamente qué modelo consumir.
        elif args.proveedor == "openrouter":
            # Falla con ayuda CLI cuando falta el ID del modelo requerido por la ruta remota.
            if not args.modelo:
                p.error("--modelo es obligatorio para OpenRouter")
            # Envía prompt y opcional few-shot por el servicio remoto elegido.
            result = chat_openrouter(row["incidencia"], args.modelo, args.fewshot)
        else:
            # Falla con ayuda CLI cuando falta el ID del modelo requerido por la ruta remota.
            if not args.modelo:
                p.error("--modelo es obligatorio para Google")
            # Consulta el recurso Google explícito, que puede identificar un modelo base o ajustado.
            result = chat_google(row["incidencia"], args.modelo)
    # Evidencia de un caso con entrada, objetivo, respuesta original, fecha y valoración.
    record = {
        "id": row["id"],
        "incidencia": row["incidencia"],
        "objetivo": row["objetivo"],
        "fecha": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **result,
        "valoracion": validate_answer(result["respuesta_original"], row["objetivo"]),
    }
    # Guardar es opcional; cuando se pide, se conserva una evidencia nueva sin sobrescribir otra.
    if args.salida:
        # Ruta concreta de lectura/escritura; el código utiliza Path para gestionar carpetas y archivos.
        path = Path(args.salida)
        # Comprueba presencia de archivo/carpeta antes de leerla o escribir; conserva la evidencia anterior.
        if path.exists():
            raise ValueError("La salida ya existe")
        # Crea la carpeta necesaria; parents/exist_ok controlan subcarpetas y rutas ya existentes.
        # Escribe el contenido legible en disco para conservar configuración o resultados de este paso.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    # Muestra el resultado/estado de este paso en la terminal; las claves no se añaden al informe.
    print(json.dumps(record, ensure_ascii=False, indent=2))


# Punto de entrada: ejecutar como script llama main; importar el módulo solo define sus funciones.
# Invoca el punto de entrada únicamente cuando se ejecuta este archivo como programa.
if __name__ == "__main__":
    main()
