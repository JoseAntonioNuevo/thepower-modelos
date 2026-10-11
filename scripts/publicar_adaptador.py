# GUÍA DE LECTURA · scripts/publicar_adaptador.py
# Publica un adaptador local en un repositorio PRIVADO NUEVO del usuario autenticado.
# Sube solo los archivos permitidos y conserva el commit devuelto por el Hub.
# Descarga ese commit, recarga base + adaptador y hace una consulta de comprobación.
# Esta acción escribe en Hugging Face; no contrata un endpoint de inferencia.

# Hace visible la raíz del proyecto al ejecutar un archivo desde scripts/.
import _bootstrap  # noqa: F401

# Permite leer opciones de la terminal y generar ayuda del comando.
# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
import argparse, json

# Path permite construir rutas, comprobar archivos y leer/escribir sin concatenar separadores a mano.
from pathlib import Path

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.common import ROOT


# Verifica usuario, crea repo privado nuevo, publica y prueba recarga por commit.
def main():
    # Parser de argumentos: describe las opciones que admite este comando.
    p = argparse.ArgumentParser()
    # Opción de terminal '--repo': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    p.add_argument("--repo", required=True, help="usuario/repo privado del curso")
    # Opción de terminal '--adaptador': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    p.add_argument("--adaptador", default=str(ROOT / "modelos/lora-soporte"))
    # Argumentos de la terminal convertidos a atributos como args.salida o args.modelo.
    args = p.parse_args()
    # Exige un repo en formato usuario/nombre para identificar quién será su propietario.
    if "/" not in args.repo:
        p.error("Indica usuario/repo")
    # Descarga checkpoints o publica archivos del adaptador en el Hub.
    from huggingface_hub import HfApi, snapshot_download

    # Cliente autenticado del Hub que crea el repositorio y sube el adaptador.
    # Identidad devuelta por Hugging Face; se contrasta con el propietario del repo solicitado.
    api = HfApi()
    who = api.whoami()
    # Validación previa: Usa el usuario autenticado; organizaciones requieren revisión explícita.
    if args.repo.split("/")[0] != who["name"]:
        raise ValueError(
            "Usa el usuario autenticado; organizaciones requieren revisión explícita"
        )
    # Nombre nuevo: no modifica repositorios previos.
    # Crea un repo privado nuevo; exist_ok=False conserva cualquier repositorio que ya exista.
    api.create_repo(args.repo, private=True, exist_ok=False)
    # Registro devuelto/creado para identificar la ejecución o el commit publicado.
    receipt = api.upload_folder(
        repo_id=args.repo,
        folder_path=args.adaptador,
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "README.md",
            "entrenamiento.json",
        ],
        commit_message="Adaptador docente S7 con datos sintéticos",
    )
    # Commit exacto publicado que se recargará para comprobar el adaptador remoto.
    revision = receipt.oid
    # Valor local del recorrido actual; su significado concreto depende de la función.
    local = snapshot_download(
        args.repo,
        revision=revision,
        allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
    )
    # Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
    from curso.qwen import load_qwen, generate

    # Recarga la base original con los archivos descargados del commit recién publicado.
    model, tok, device = load_qwen(local)
    # Hace una consulta de desarrollo para comprobar que el adaptador remoto realmente se puede utilizar.
    result = generate(model, tok, device, "Tengo un cargo duplicado en mi matrícula.")
    # Ruta concreta de lectura/escritura; el código utiliza Path para gestionar carpetas y archivos.
    path = ROOT / "material/evidencias/huggingface-recarga.json"
    # Escribe el contenido legible en disco para conservar configuración o resultados de este paso.
    path.write_text(
        json.dumps(
            {
                "repo": args.repo,
                "revision": revision,
                "private": True,
                "recarga": result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    # Muestra el resultado/estado de este paso en la terminal; las claves no se añaden al informe.
    print(json.dumps({"repo": args.repo, "revision": revision, "private": True}))


# Punto de entrada: ejecutar como script llama main; importar el módulo solo define sus funciones.
# Invoca el punto de entrada únicamente cuando se ejecuta este archivo como programa.
if __name__ == "__main__":
    main()
