# GUÍA DE LECTURA · scripts/resumir.py
# Convierte lotes completos en evidencia JSON, CSV y una tabla Markdown.
# Todos los casos permanecen en el denominador, incluidos errores e inválidos.
# Antes de combinar, comprueba IDs, partición, tamaño y manifiesto del experimento.
# El CSV neutraliza fórmulas; el JSON conserva exactamente el texto original.

# Hace visible la raíz del proyecto al ejecutar un archivo desde scripts/.
import _bootstrap  # noqa: F401

# Permite leer opciones de la terminal y generar ayuda del comando.
# Escribe/lee tablas CSV con comillas y separadores correctos.
# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
# Calcula medianas de tiempos conocidos sin convertir ausencias en cero.
import argparse, csv, json, statistics

# Path permite construir rutas, comprobar archivos y leer/escribir sin concatenar separadores a mano.
from pathlib import Path

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.common import ROOT, read_jsonl


# Seis condiciones del ejercicio; Gemini/OpenRouter se presenta como demo adicional.
CONDITIONS = (
    "qwen_base",
    "qwen_fewshot",
    "qwen_lora",
    "gemini_base",
    "gemini_ajustado",
    "claude_prompt",
)


# Neutraliza texto que una hoja de cálculo podría interpretar como fórmula.
# Esta representación protege el CSV; la evidencia JSON se conserva sin alteración.
def csv_text(value):
    """La evidencia JSON permanece exacta; el CSV trata las fórmulas como texto."""
    # Detecta prefijos de fórmula y caracteres de control antes de escribir texto en CSV.
    if isinstance(value, str) and (
        value.startswith(("\t", "\r", "\n"))
        or value.lstrip().startswith(("=", "+", "-", "@"))
    ):
        return "'" + value
    return value


# Cuenta aciertos y errores, calcula mediana de tiempos conocidos y suma coste conocido.
# Si falta cualquier coste, devuelve None en lugar de inventar un total.
def summarize(rows):
    # Número total de casos del lote; los inválidos siguen en el denominador.
    n = len(rows)
    # Devuelve conteos y métricas disponibles; no convierte medidas desconocidas en cero.
    return {
        "casos": n,
        **{
            key: sum(bool(r["valoracion"][key]) for r in rows)
            for key in (
                "json_valido",
                "esquema_valido",
                "categoria_correcta",
                "prioridad_correcta",
                "ambas_correctas",
            )
        },
        "errores": sum(bool(r.get("error")) for r in rows),
        "latencia_mediana_s": statistics.median(
            r["latencia_s"] for r in rows if r.get("latencia_s") is not None
        )
        if any(r.get("latencia_s") is not None for r in rows)
        else None,
        "coste_usd": sum(r["coste_usd"] for r in rows)
        if rows and all(r.get("coste_usd") is not None for r in rows)
        else None,
    }


# Verifica lotes comparables y escribe registros JSON, resumen, CSV y tabla Markdown.
def main():
    # Parser de argumentos: describe las opciones que admite este comando.
    # Opción de terminal '--carpeta': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    p = argparse.ArgumentParser()
    p.add_argument("--carpeta", default=str(ROOT / "resultados"))
    # Opción de terminal '--destino': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    # Argumentos de la terminal convertidos a atributos como args.salida o args.modelo.
    p.add_argument("--destino", default=str(ROOT / "material/evidencias"))
    args = p.parse_args()
    # Carpeta que se recorre para leer lotes o generar una salida temporal.
    # Destino donde esta ejecución guardará sus archivos.
    # Crea la carpeta necesaria; parents/exist_ok controlan subcarpetas y rutas ya existentes.
    folder = Path(args.carpeta)
    output = Path(args.destino)
    output.mkdir(parents=True, exist_ok=True)
    # Resultados originales reunidos para generar JSON, CSV y resumen.
    # Resumen por condición; distingue resultados medidos y condiciones pendientes.
    records = []
    report = {}
    # Procesa cada condición por separado para no mezclar sus casos y métricas.
    for cond in CONDITIONS + ("gemini_openrouter_demo",):
        # Ruta concreta de lectura/escritura; el código utiliza Path para gestionar carpetas y archivos.
        path = folder / f"{cond}.jsonl"
        # Registros de la partición o lote actual, conservados como una lista de diccionarios.
        rows = read_jsonl(path) if path.exists() else []
        # Solo resume un lote si existe evidencia; las condiciones ausentes siguen pendientes.
        if rows:
            # Validación previa: IDs repetidos en evaluación.
            if len({r["id"] for r in rows}) != len(rows):
                raise ValueError("IDs repetidos en evaluación")
            # Exige un lote completo de 30 casos; no presenta un subconjunto como el resultado final.
            if len(rows) != 30:
                raise ValueError(f"Lote incompleto {cond}: {len(rows)}")
            # Validación previa: Mezcla desarrollo y test.
            if any(r["split"] != "test" for r in rows):
                raise ValueError("Mezcla desarrollo y test")
            # Validación previa: Configuración cambió durante lote.
            if len({json.dumps(r["experimento"], sort_keys=True) for r in rows}) != 1:
                raise ValueError("Configuración cambió durante lote")
            # Calcula estadísticas de la condición y conserva también sus registros originales.
            report[cond] = summarize(rows)
            records += rows
        # Marca ausencia de ejecución explícitamente; cero casos no se presenta como cero errores.
        else:
            report[cond] = {"casos": 0, "estado": "PENDIENTE; no hay ejecución"}
    # Compara manifiestos entre condiciones para impedir mezclar protocolos distintos en una tabla.
    manifests = {json.dumps(r["experimento"], sort_keys=True) for r in records}
    # Validación previa: Lotes con distintos datos/protocolo.
    if len(manifests) > 1:
        raise ValueError("Lotes con distintos datos/protocolo")
    # Escribe el contenido legible en disco para conservar configuración o resultados de este paso.
    (output / "resultados.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    )
    # Escribe el contenido legible en disco para conservar configuración o resultados de este paso.
    (output / "resumen.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    # Columnas del CSV, separando metadatos, respuesta original y banderas del validador.
    fields = [
        "id",
        "condicion",
        "model",
        "provider",
        "latencia_s",
        "tokens_entrada",
        "tokens_salida",
        "coste_usd",
        "error",
        "respuesta_original",
        "json_valido",
        "esquema_valido",
        "categoria_correcta",
        "prioridad_correcta",
        "ambas_correctas",
    ]
    # El contexto abre el archivo y lo cierra automáticamente al terminar.
    with (output / "resultados.csv").open("w", newline="") as f:
        # Crea columnas explícitas y escribe cabecera; CSV conserva claves de valoración y metadatos.
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        # Escribe cada registro neutralizando fórmulas solo en la representación CSV.
        for row in records:
            writer.writerow(
                {k: csv_text(row.get(k, row["valoracion"].get(k))) for k in fields}
            )
    # Líneas del informe Markdown que se guardará al finalizar.
    lines = [
        "# Comparación S7/S8",
        "",
        "Muestra sintética de 30 casos por condición. Sin reparación ni reintentos correctores.",
        "",
        "| Condición | Casos | JSON | Categoría | Prioridad | Ambas | Mediana s | Coste USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    # Procesa cada condición por separado para no mezclar sus casos y métricas.
    for cond in CONDITIONS:
        # Recupera las estadísticas que se van a presentar en una fila de Markdown.
        r = report[cond]
        # Muestra pendiente y No disponible cuando no hay ejecución, y pasa a la siguiente condición.
        if not r["casos"]:
            lines.append(f"| {cond} | Pendiente | - | - | - | - | - | No disponible |")
            continue
        # Función auxiliar de presentación: muestra desconocidos como No disponible en la tabla.
        fmt = lambda k: str(r[k]) if r[k] is not None else "No disponible"
        # Construye la fila visible con conteos, mediana y coste disponible de la condición.
        lines.append(
            "| "
            + " | ".join(
                [cond, str(r["casos"])]
                + [
                    fmt(k)
                    for k in (
                        "json_valido",
                        "categoria_correcta",
                        "prioridad_correcta",
                        "ambas_correctas",
                        "latencia_mediana_s",
                        "coste_usd",
                    )
                ]
            )
            + " |"
        )
    # Añade el límite docente: Gemini/OpenRouter no sustituye el par Google base/ajustado.
    lines += [
        "",
        "Las filas sin ejecución permanecen pendientes. Gemini por OpenRouter es una condición adicional;",
        "no sustituye la pareja Google base/ajustado.",
    ]
    # Incluye por separado la condición adicional solo si tiene resultados medidos.
    if report["gemini_openrouter_demo"]["casos"]:
        lines += [
            "",
            "Resultado adicional Gemini por OpenRouter: "
            + json.dumps(report["gemini_openrouter_demo"], ensure_ascii=False),
        ]
    # Escribe el contenido legible en disco para conservar configuración o resultados de este paso.
    # Muestra el resultado/estado de este paso en la terminal; las claves no se añaden al informe.
    (output / "comparacion.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report))


# Punto de entrada: ejecutar como script llama main; importar el módulo solo define sus funciones.
# Invoca el punto de entrada únicamente cuando se ejecuta este archivo como programa.
if __name__ == "__main__":
    main()
