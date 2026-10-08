import _bootstrap  # noqa: F401
import argparse,json,time
from pathlib import Path
from curso.common import ROOT,read_jsonl,validate_answer,assert_frozen


def main():
    p=argparse.ArgumentParser(description="Un caso de desarrollo; no usa el test.")
    p.add_argument("--proveedor",choices=("local","openrouter","qwen","google"),required=True)
    p.add_argument("--modelo");p.add_argument("--caso",default="DE-01")
    p.add_argument("--local-api",choices=("native","openai"),default="native")
    p.add_argument("--fewshot",action="store_true");p.add_argument("--adaptador")
    p.add_argument("--salida")
    args=p.parse_args()
    row=next(x for x in read_jsonl(ROOT/"datos/desarrollo.jsonl") if x["id"]==args.caso)
    if args.proveedor=="qwen":
        assert_frozen()
        from curso.qwen import load_qwen,generate
        model,tok,device=load_qwen(args.adaptador)
        result=generate(model,tok,device,row["incidencia"],args.fewshot)
    else:
        from curso.remote import chat_local,chat_openrouter,chat_google
        if args.proveedor=="local": result=chat_local(row["incidencia"],args.modelo,args.fewshot,args.local_api=="native")
        elif args.proveedor=="openrouter":
            if not args.modelo: p.error("--modelo es obligatorio para OpenRouter")
            result=chat_openrouter(row["incidencia"],args.modelo,args.fewshot)
        else:
            if not args.modelo:p.error("--modelo es obligatorio para Google")
            result=chat_google(row["incidencia"],args.modelo)
    record={"id":row["id"],"incidencia":row["incidencia"],"objetivo":row["objetivo"],
            "fecha":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),**result,
            "valoracion":validate_answer(result["respuesta_original"],row["objetivo"])}
    if args.salida:
        path=Path(args.salida)
        if path.exists():raise ValueError("La salida ya existe")
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
