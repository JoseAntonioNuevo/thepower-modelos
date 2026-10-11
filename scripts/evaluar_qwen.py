# GUÍA DE LECTURA · scripts/evaluar_qwen.py
# Ejecuta un lote local en una condición: básico, few-shot o LoRA.
# Cada caso conserva entrada, objetivo, salida, configuración y valoración.
# Una excepción queda registrada y cuenta dentro del lote; no se reintenta para mejorar cifras.
# El archivo se actualiza tras cada caso para conservar lo ya completado.

# Hace visible la raíz del proyecto al ejecutar un archivo desde scripts/.
import _bootstrap  # noqa: F401

# Permite leer opciones de la terminal y generar ayuda del comando.
import argparse

# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
import json

# Mide duración y prepara fechas; perf_counter/monotonic sirven para intervalos, no fechas del calendario.
import time

# Path permite construir rutas, comprobar archivos y leer/escribir sin concatenar separadores a mano.
from pathlib import Path

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.common import ROOT, assert_frozen, read_jsonl, validate_answer, write_jsonl


# Recorre una partición con la condición Qwen elegida y guarda originales y errores.
def main():
    # Parser de argumentos: describe las opciones que admite este comando.
    p = argparse.ArgumentParser()
    # Opción de terminal '--condicion': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    p.add_argument(
        "--condicion", choices=("qwen_base", "qwen_fewshot", "qwen_lora"), required=True
    )
    # Opción de terminal '--split': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    p.add_argument(
        "--split", choices=("desarrollo", "validacion", "test"), default="desarrollo"
    )
    # Opción de terminal '--adaptador': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    p.add_argument("--adaptador", default=str(ROOT / "modelos/lora-soporte"))
    # Opción de terminal '--salida': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    p.add_argument("--salida", required=True)
    # Argumentos de la terminal convertidos a atributos como args.salida o args.modelo.
    # Manifiesto verificado que acompaña los resultados y conserva el protocolo exacto.
    args = p.parse_args()
    frozen = assert_frozen()
    # Comprueba presencia de archivo/carpeta antes de leerla o escribir; conserva la evidencia anterior.
    if Path(args.salida).exists():
        raise ValueError("Salida existe; elige otra para conservar evidencia")
    # Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
    from curso.qwen import load_qwen, generate

    # Carga el adaptador solo para qwen_lora; las otras dos condiciones mantienen la base original.
    model, tokenizer, device = load_qwen(
        args.adaptador if args.condicion == "qwen_lora" else None
    )
    # Resultados acumulados, incluyendo errores, para conservar un caso por entrada.
    results = []
    # Recorre cada registro de la partición elegida; conserva el ID para relacionar entradas y resultados.
    for row in read_jsonl(ROOT / f"datos/{args.split}.jsonl"):
        try:
            # Destino donde esta ejecución guardará sus archivos.
            output = generate(
                model,
                tokenizer,
                device,
                row["incidencia"],
                args.condicion == "qwen_fewshot",
            )
            # Tipo de fallo del caso; None indica que la petición/generación terminó sin esa excepción.
            error = None
        except Exception as exc:
            # Destino donde esta ejecución guardará sus archivos.
            # Tipo de fallo del caso; None indica que la petición/generación terminó sin esa excepción.
            output = {"respuesta_original": None, "latencia_s": None, "coste_usd": None}
            error = type(exc).__name__
        # Conserva un resultado por caso incluso si hubo excepción; el objetivo permite calcular aciertos.
        results.append(
            {
                "id": row["id"],
                "condicion": args.condicion,
                "split": args.split,
                "incidencia": row["incidencia"],
                "objetivo": row["objetivo"],
                "fecha": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "experimento": frozen,
                "adaptador": args.adaptador if args.condicion == "qwen_lora" else None,
                **output,
                "error": error,
                "valoracion": validate_answer(
                    output.get("respuesta_original"), row["objetivo"]
                ),
            }
        )
        # Guarda los registros del paso como JSONL, una fila por ejemplo y sin reparar respuestas.
        write_jsonl(args.salida, results)
        # Muestra el resultado/estado de este paso en la terminal; las claves no se añaden al informe.
        print(row["id"], args.condicion, "error=" + str(error), flush=True)


# Punto de entrada: ejecutar como script llama main; importar el módulo solo define sus funciones.
# Invoca el punto de entrada únicamente cuando se ejecuta este archivo como programa.
if __name__ == "__main__":
    main()
