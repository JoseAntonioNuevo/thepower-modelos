# GUÍA DE LECTURA · scripts/convertir_gemini.py
# Convierte las mismas particiones de S7 al JSONL requerido por Google.
# Conserva incidencia y respuesta objetivo y registra los IDs en un archivo aparte.
# Convertir datos no los sube a Google ni inicia un entrenamiento.
# Test se convierte para evaluar después, separado de train y validación.

# Hace visible la raíz del proyecto al ejecutar un archivo desde scripts/.
import _bootstrap  # noqa: F401

# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
import json

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.common import ROOT, read_jsonl, write_jsonl, assert_frozen, digest

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.remote import gemini_record


# Escribe el JSONL convertido de cada partición y su correspondencia de IDs.
def main():
    # Comprueba integridad antes de convertir e inicia la correspondencia de IDs.
    assert_frozen()
    mapping = {}
    # Convierte cada partición por separado; ninguna línea de test entra en train.
    for split in ("train", "validacion", "test"):
        # Registros de la partición o lote actual, conservados como una lista de diccionarios.
        rows = read_jsonl(ROOT / f"datos/{split}.jsonl")
        # Ruta concreta de lectura/escritura; el código utiliza Path para gestionar carpetas y archivos.
        path = ROOT / f"datos/gemini/{split}.jsonl"
        # Guarda los registros del paso como JSONL, una fila por ejemplo y sin reparar respuestas.
        write_jsonl(path, [gemini_record(row) for row in rows])
        # Relaciona orden de líneas con IDs originales y calcula el hash del archivo convertido.
        mapping[split] = {
            "ids_por_linea": [r["id"] for r in rows],
            "sha256": digest(path),
        }
    # Escribe el contenido legible en disco para conservar configuración o resultados de este paso.
    (ROOT / "datos/gemini/ids.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n"
    )
    # Muestra el resultado/estado de este paso en la terminal; las claves no se añaden al informe.
    print(
        "JSONL convertido; test separado de los archivos de entrenamiento y validación"
    )


# Punto de entrada: ejecutar como script llama main; importar el módulo solo define sus funciones.
# Invoca el punto de entrada únicamente cuando se ejecuta este archivo como programa.
if __name__ == "__main__":
    main()
