import _bootstrap  # noqa: F401
import argparse
import json
import time
from pathlib import Path
from curso.common import ROOT,assert_frozen,read_jsonl,validate_answer,write_jsonl


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--condicion",choices=("qwen_base","qwen_fewshot","qwen_lora"),required=True)
    p.add_argument("--split",choices=("desarrollo","validacion","test"),default="desarrollo")
    p.add_argument("--adaptador",default=str(ROOT/"modelos/lora-soporte"))
    p.add_argument("--salida",required=True)
    args=p.parse_args();frozen=assert_frozen()
    if Path(args.salida).exists(): raise ValueError("Salida existe; elige otra para conservar evidencia")
    from curso.qwen import load_qwen,generate
    model,tokenizer,device=load_qwen(args.adaptador if args.condicion=="qwen_lora" else None)
    results=[]
    for row in read_jsonl(ROOT/f"datos/{args.split}.jsonl"):
        try:
            output=generate(model,tokenizer,device,row["incidencia"],args.condicion=="qwen_fewshot")
            error=None
        except Exception as exc:
            output={"respuesta_original":None,"latencia_s":None,"coste_usd":None};error=type(exc).__name__
        results.append({"id":row["id"],"condicion":args.condicion,"split":args.split,
            "incidencia":row["incidencia"],"objetivo":row["objetivo"],"fecha":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "experimento":frozen,"adaptador":args.adaptador if args.condicion=="qwen_lora" else None,
            **output,"error":error,"valoracion":validate_answer(output.get("respuesta_original"),row["objetivo"])})
        write_jsonl(args.salida,results)
        print(row["id"],args.condicion,"error="+str(error),flush=True)


if __name__=="__main__": main()
