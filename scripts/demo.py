# GUÍA DE LECTURA · scripts/demo.py
# Ejecuta una consulta de desarrollo y conserva la respuesta tal como llegó.
# --proveedor elige el recorrido: qwen (Transformers) o una ruta de curso.remote.
# Sin --proveedor se usa LLM_PROVIDER/LLM_MODEL de .env (por defecto, Claude por Anthropic).
# --adaptador activa LoRA con Qwen.
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

# Rutas remotas/locales disponibles; la clave coincide con los valores de LLM_PROVIDER.
from curso.remote import PROVEEDORES


# Lee el caso y los argumentos, llama al proveedor y registra respuesta/valoración.
def main():
    # Parser de argumentos: describe las opciones que admite este comando.
    p = argparse.ArgumentParser(description="Un caso de desarrollo; no usa el test.")
    # Opción de terminal '--proveedor': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    # Sin valor, la ruta sale de LLM_PROVIDER en .env.
    p.add_argument("--proveedor", choices=("qwen", *PROVEEDORES))
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
    # Ruta LM Studio: --local-api decide petición nativa o compatible.
    elif args.proveedor == "local":
        # Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
        from curso.remote import chat_local

        result = chat_local(
            row["incidencia"], args.modelo, args.fewshot, args.local_api == "native"
        )
    else:
        # Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
        from curso.remote import chat

        # Ruta de --proveedor o de .env; --modelo sustituye a LLM_MODEL.
        try:
            result = chat(row["incidencia"], args.proveedor, args.modelo, args.fewshot)
        # Configuración incompleta (modelo, clave o URL): ayuda CLI en lugar de una traza.
        except ValueError as exc:
            p.error(str(exc))
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
