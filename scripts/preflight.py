# GUÍA DE LECTURA · scripts/preflight.py
# Comprueba datos, hashes, dependencias y presencia del adaptador antes de clase.
# Informa si hay configuración, pero nunca imprime claves ni hace inferencia.
# Sin --local no consulta servidores; con --local solo lee el catálogo de modelos.
# Un preflight correcto no demuestra que un proveedor remoto o el entrenamiento funcionen.

# Hace visible la raíz del proyecto al ejecutar un archivo desde scripts/.
import _bootstrap  # noqa: F401

# Permite leer opciones de la terminal y generar ayuda del comando.
# Comprueba si un paquete está disponible sin cargar el modelo ni llamar al proveedor.
# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
# Lee configuración del entorno y ajusta opciones del proceso.
# Identifica versión de Python y entorno de ejecución para el registro.
import argparse, importlib.util, json, os, platform

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.common import ROOT, validate_splits, assert_frozen

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.remote import load_config, configured_provider, configured_model

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.http import bounded_json, model_ids


# Lee --local, reúne comprobaciones y muestra un informe sin valores secretos.
def main():
    # Parser de argumentos: describe las opciones que admite este comando.
    # Opción de terminal '--local': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    # Argumentos de la terminal convertidos a atributos como args.salida o args.modelo.
    # Carga configuración del entorno antes de comprobar opciones o crear una petición.
    p = argparse.ArgumentParser()
    p.add_argument("--local", action="store_true")
    args = p.parse_args()
    load_config()
    # Reúne versión, integridad, paquetes y presencia de configuración; solo publica booleanos de credenciales.
    result = {
        "python": platform.python_version(),
        "splits": validate_splits(),
        "experimento_congelado": bool(assert_frozen()),
        "dependencias": {
            m: importlib.util.find_spec(m) is not None
            for m in (
                "requests",
                "anthropic",
                "transformers",
                "torch",
                "trl",
                "peft",
                "datasets",
                "google.genai",
            )
        },
        # Ruta de LLM_PROVIDER/LLM_MODEL que usarán demo y evaluación remota sin --proveedor.
        "ruta_remota": {
            "proveedor": configured_provider(),
            "modelo": configured_model(),
        },
        # Presencia de credenciales por proveedor: solo booleanos, nunca valores.
        "credenciales": {
            name: bool(os.getenv(var))
            for name, var in (
                ("anthropic", "ANTHROPIC_API_KEY"),
                ("openai", "OPENAI_API_KEY"),
                ("openrouter", "OPENROUTER_API_KEY"),
                ("gemini", "GEMINI_API_KEY"),
                ("compatible", "COMPAT_BASE_URL"),
            )
        },
        "google_proyecto_curso_configurado": bool(os.getenv("GOOGLE_CLOUD_PROJECT")),
        "adaptador_local": (
            ROOT / "modelos/lora-soporte/adapter_model.safetensors"
        ).exists(),
        "datos_sinteticos": True,
        "conexion_remota": "No probada; preflight no consume inferencia",
    }
    # --local añade una lectura de catálogo HTTP; sin esta opción el preflight no consulta LM Studio.
    if args.local:
        # Cliente HTTP; las peticiones se realizan al llamarlo, no por el simple import.
        import requests

        # Dirección de la petición o del servidor local que se va a consultar.
        url = (
            os.getenv("LOCAL_BASE_URL", "http://localhost:1234/v1").rstrip("/")
            + "/models"
        )
        try:
            # Respuesta HTTP que se inspeccionará y cerrará después de leerla.
            response = requests.get(url, timeout=5, stream=True, allow_redirects=False)
            try:
                # Comprueba el estado HTTP antes de tratar el cuerpo como una respuesta válida del modelo.
                if response.status_code != 200:
                    raise ValueError("Catálogo HTTP no disponible")
                # Detiene la lectura si la API local devolvió un estado de error.
                response.raise_for_status()
                # Añade IDs de modelos válidos al informe; limita a diez segundos el recorrido de bloques.
                result["modelos_locales"] = model_ids(
                    bounded_json(response, max_seconds=10)
                )
            # Cierra el recurso abierto; libera socket/conexión aunque el paso anterior haya fallado.
            finally:
                response.close()
        # Representa fallo HTTP/JSON como disponibilidad desconocida, sin imprimir cuerpos o claves.
        except (requests.RequestException, ValueError):
            result["modelos_locales"] = (
                "Servidor local no disponible o respuesta inválida"
            )
    # Muestra el resultado/estado de este paso en la terminal; las claves no se añaden al informe.
    print(json.dumps(result, ensure_ascii=False, indent=2))


# Punto de entrada: ejecutar como script llama main; importar el módulo solo define sus funciones.
# Invoca el punto de entrada únicamente cuando se ejecuta este archivo como programa.
if __name__ == "__main__":
    main()
