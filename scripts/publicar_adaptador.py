import _bootstrap  # noqa: F401
import argparse,json
from pathlib import Path
from curso.common import ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",required=True,help="usuario/repo privado del curso")
    p.add_argument("--adaptador",default=str(ROOT/"modelos/lora-soporte"))
    args=p.parse_args()
    if '/' not in args.repo:p.error("Indica usuario/repo")
    from huggingface_hub import HfApi,snapshot_download
    api=HfApi();who=api.whoami()
    if args.repo.split('/')[0]!=who['name']:raise ValueError("Usa el usuario autenticado; organizaciones requieren revisión explícita")
    # Nombre nuevo: no modifica repositorios previos.
    api.create_repo(args.repo,private=True,exist_ok=False)
    receipt=api.upload_folder(repo_id=args.repo,folder_path=args.adaptador,
        allow_patterns=["adapter_config.json","adapter_model.safetensors","README.md","entrenamiento.json"],
        commit_message="Adaptador docente S7 con datos sintéticos")
    revision=receipt.oid
    local=snapshot_download(args.repo,revision=revision,allow_patterns=["adapter_config.json","adapter_model.safetensors"])
    from curso.qwen import load_qwen,generate
    model,tok,device=load_qwen(local)
    result=generate(model,tok,device,"Tengo un cargo duplicado en mi matrícula.")
    path=ROOT/"material/evidencias/huggingface-recarga.json"
    path.write_text(json.dumps({"repo":args.repo,"revision":revision,"private":True,"recarga":result},ensure_ascii=False,indent=2))
    print(json.dumps({"repo":args.repo,"revision":revision,"private":True}))


if __name__=="__main__":main()
