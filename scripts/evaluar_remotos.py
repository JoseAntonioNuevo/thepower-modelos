import _bootstrap  # noqa: F401
import argparse,json,time
from pathlib import Path
from curso.common import ROOT,read_jsonl,assert_frozen,validate_answer,write_jsonl
from curso.remote import chat_openrouter,chat_google


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--condicion",choices=("claude_prompt","gemini_base","gemini_ajustado","gemini_openrouter_demo"),required=True)
    p.add_argument("--modelo");p.add_argument("--salida",required=True)
    p.add_argument("--split",choices=("desarrollo","validacion","test"),default="desarrollo")
    args=p.parse_args();frozen=assert_frozen()
    if Path(args.salida).exists():raise ValueError("Salida existente; conservar resultados")
    if args.condicion=="gemini_ajustado" and not args.modelo:p.error("Indica el recurso de modelo ajustado")
    model=args.modelo or {"claude_prompt":"anthropic/claude-haiku-4.5","gemini_base":"gemini-3.1-flash-lite",
        "gemini_openrouter_demo":"google/gemini-3.1-flash-lite"}.get(args.condicion)
    results=[]
    for row in read_jsonl(ROOT/f"datos/{args.split}.jsonl"):
        try:
            result=(chat_openrouter(row["incidencia"],model,args.condicion=="claude_prompt")
                    if args.condicion in ("claude_prompt","gemini_openrouter_demo") else chat_google(row["incidencia"],model))
            error=None
        except Exception as exc:
            result={"respuesta_original":None,"latencia_s":None,"coste_usd":None};error=type(exc).__name__
        results.append({"id":row["id"],"condicion":args.condicion,"split":args.split,"experimento":frozen,
            "incidencia":row["incidencia"],"objetivo":row["objetivo"],**result,"error":error,
            "fecha":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "valoracion":validate_answer(result.get("respuesta_original"),row["objetivo"])})
        write_jsonl(args.salida,results);print(row["id"],args.condicion,"error="+str(error),flush=True)


if __name__=="__main__":main()
