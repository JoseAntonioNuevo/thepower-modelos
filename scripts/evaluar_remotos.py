# GUÍA DE LECTURA · scripts/evaluar_remotos.py
# Evalúa modelos remotos con los mismos datos y validador de S7.
# claude_prompt y remoto_basico usan la ruta de .env (LLM_PROVIDER/LLM_MODEL) o de
# --proveedor/--modelo: por defecto, Claude Haiku 5.5 por la API de Anthropic.
# Gemini base/ajustado van por Google Cloud; la condición ajustada necesita su recurso.
# Un fallo permanece en los resultados y no se sustituye por otro modelo.
# Las llamadas remotas pueden consumir presupuesto; no se hacen al importar el archivo.

# Hace visible la raíz del proyecto al ejecutar un archivo desde scripts/.
import _bootstrap  # noqa: F401

# Permite leer opciones de la terminal y generar ayuda del comando.
# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
# Mide duración y prepara fechas; perf_counter/monotonic sirven para intervalos, no fechas del calendario.
import argparse, json, time

# Path permite construir rutas, comprobar archivos y leer/escribir sin concatenar separadores a mano.
from pathlib import Path

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.common import ROOT, read_jsonl, assert_frozen, validate_answer, write_jsonl

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.remote import chat, require_credentials, route


# Condiciones con ruta fija: el par Gemini del ejercicio y la demo histórica por OpenRouter.
FIXED_ROUTES = {
    "gemini_base": ("google", "gemini-3.1-flash-lite"),
    "gemini_ajustado": ("google", None),
    "gemini_openrouter_demo": ("openrouter", "google/gemini-3.1-flash-lite"),
}


# Selecciona ruta y modelo remotos, recorre los casos y conserva un resultado por caso.
def main():
    # Parser de argumentos: describe las opciones que admite este comando.
    p = argparse.ArgumentParser()
    # Opción de terminal '--condicion': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    p.add_argument(
        "--condicion",
        choices=(
            "claude_prompt",
            "remoto_basico",
            "gemini_base",
            "gemini_ajustado",
            "gemini_openrouter_demo",
        ),
        required=True,
    )
    # Opción de terminal '--modelo': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    # Opción de terminal '--salida': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    # Opción de terminal '--proveedor': ruta de claude_prompt/remoto_basico; sin ella se usa .env.
    p.add_argument("--proveedor")
    p.add_argument("--modelo")
    p.add_argument("--salida", required=True)
    # Opción de terminal '--split': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    p.add_argument(
        "--split", choices=("desarrollo", "validacion", "test"), default="desarrollo"
    )
    # Argumentos de la terminal convertidos a atributos como args.salida o args.modelo.
    # Manifiesto verificado que acompaña los resultados y conserva el protocolo exacto.
    args = p.parse_args()
    frozen = assert_frozen()
    # Comprueba presencia de archivo/carpeta antes de leerla o escribir; conserva la evidencia anterior.
    if Path(args.salida).exists():
        raise ValueError("Salida existente; conservar resultados")
    # El recurso ajustado no se puede deducir: debe proceder de un entrenamiento Google real.
    if args.condicion == "gemini_ajustado" and not args.modelo:
        p.error("Indica el recurso de modelo ajustado")
    # Ruta fija para Gemini; las demás condiciones admiten cualquier proveedor configurado.
    if args.condicion in FIXED_ROUTES:
        provider, default_model = FIXED_ROUTES[args.condicion]
        # Validación previa: estas condiciones no cambian de proveedor.
        if args.proveedor and args.proveedor != provider:
            p.error(f"{args.condicion} usa siempre la ruta {provider}")
        provider, model = provider, args.modelo or default_model
    else:
        provider, model = args.proveedor, args.modelo
    # Resuelve y valida la ruta antes del lote: una configuración incompleta no genera 30 errores.
    try:
        provider, model = route(provider, model)
        require_credentials(provider)
    except ValueError as exc:
        p.error(str(exc))
    # claude_prompt recibe los cuatro ejemplos few-shot fijados; el resto, el prompt básico.
    fewshot = args.condicion == "claude_prompt"
    # Resultados acumulados, incluyendo errores, para conservar un caso por entrada.
    results = []
    # Recorre cada registro de la partición elegida; conserva el ID para relacionar entradas y resultados.
    for row in read_jsonl(ROOT / f"datos/{args.split}.jsonl"):
        try:
            # Misma ruta y modelo en todos los casos del lote.
            result = chat(row["incidencia"], provider, model, fewshot)
            # Tipo de fallo del caso; None indica que la petición/generación terminó sin esa excepción.
            error = None
        except Exception as exc:
            # Conserva el fallo sin respuesta/tiempo/coste inventados; el tipo de excepción identifica el error.
            result = {
                "respuesta_original": None,
                "latencia_s": None,
                "coste_usd": None,
                "model": model,
                "ruta": provider,
            }
            error = type(exc).__name__
        # Guarda original, objetivo, fallo y fecha con el mismo manifiesto utilizado para el resto del lote.
        results.append(
            {
                "id": row["id"],
                "condicion": args.condicion,
                "split": args.split,
                "experimento": frozen,
                "incidencia": row["incidencia"],
                "objetivo": row["objetivo"],
                **result,
                "error": error,
                "fecha": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "valoracion": validate_answer(
                    result.get("respuesta_original"), row["objetivo"]
                ),
            }
        )
        # Guarda los registros del paso como JSONL, una fila por ejemplo y sin reparar respuestas.
        # Muestra el resultado/estado de este paso en la terminal; las claves no se añaden al informe.
        write_jsonl(args.salida, results)
        print(row["id"], args.condicion, "error=" + str(error), flush=True)


# Punto de entrada: ejecutar como script llama main; importar el módulo solo define sus funciones.
# Invoca el punto de entrada únicamente cuando se ejecuta este archivo como programa.
if __name__ == "__main__":
    main()
