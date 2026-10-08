import _bootstrap  # noqa: F401
import argparse,csv,json,statistics
from pathlib import Path
from curso.common import ROOT,read_jsonl


CONDITIONS=("qwen_base","qwen_fewshot","qwen_lora","gemini_base","gemini_ajustado","claude_prompt")


def csv_text(value):
    """La evidencia JSON permanece exacta; el CSV trata las fórmulas como texto."""
    if isinstance(value,str) and (value.startswith(("\t","\r","\n")) or value.lstrip().startswith(("=","+","-","@"))):
        return "'"+value
    return value


def summarize(rows):
    n=len(rows)
    return {"casos":n,**{key:sum(bool(r["valoracion"][key]) for r in rows) for key in
        ("json_valido","esquema_valido","categoria_correcta","prioridad_correcta","ambas_correctas")},
        "errores":sum(bool(r.get("error")) for r in rows),
        "latencia_mediana_s":statistics.median(r["latencia_s"] for r in rows if r.get("latencia_s") is not None)
             if any(r.get("latencia_s") is not None for r in rows) else None,
        "coste_usd":sum(r["coste_usd"] for r in rows) if rows and all(r.get("coste_usd") is not None for r in rows) else None}


def main():
    p=argparse.ArgumentParser();p.add_argument("--carpeta",default=str(ROOT/"resultados"))
    p.add_argument("--destino",default=str(ROOT/"material/evidencias"));args=p.parse_args()
    folder=Path(args.carpeta);output=Path(args.destino);output.mkdir(parents=True,exist_ok=True)
    records=[];report={}
    for cond in CONDITIONS+("gemini_openrouter_demo",):
        path=folder/f"{cond}.jsonl"
        rows=read_jsonl(path) if path.exists() else []
        if rows:
            if len({r['id'] for r in rows})!=len(rows):raise ValueError("IDs repetidos en evaluación")
            if len(rows)!=30:raise ValueError(f"Lote incompleto {cond}: {len(rows)}")
            if any(r["split"]!="test" for r in rows):raise ValueError("Mezcla desarrollo y test")
            if len({json.dumps(r["experimento"],sort_keys=True) for r in rows})!=1:raise ValueError("Configuración cambió durante lote")
            report[cond]=summarize(rows);records+=rows
        else:report[cond]={"casos":0,"estado":"PENDIENTE; no hay ejecución"}
    manifests={json.dumps(r["experimento"],sort_keys=True) for r in records}
    if len(manifests)>1:raise ValueError("Lotes con distintos datos/protocolo")
    (output/"resultados.json").write_text(json.dumps(records,ensure_ascii=False,indent=2)+"\n")
    (output/"resumen.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    fields=['id','condicion','model','provider','latencia_s','tokens_entrada','tokens_salida','coste_usd','error','respuesta_original',
            'json_valido','esquema_valido','categoria_correcta','prioridad_correcta','ambas_correctas']
    with (output/"resultados.csv").open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in records:writer.writerow({k:csv_text(row.get(k,row['valoracion'].get(k))) for k in fields})
    lines=['# Comparación S7/S8','', 'Muestra sintética de 30 casos por condición. Sin reparación ni reintentos correctores.',
           '', '| Condición | Casos | JSON | Categoría | Prioridad | Ambas | Mediana s | Coste USD |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for cond in CONDITIONS:
        r=report[cond]
        if not r['casos']:lines.append(f'| {cond} | Pendiente | - | - | - | - | - | No disponible |');continue
        fmt=lambda k: str(r[k]) if r[k] is not None else 'No disponible'
        lines.append('| '+ ' | '.join([cond,str(r['casos'])]+[fmt(k) for k in
            ('json_valido','categoria_correcta','prioridad_correcta','ambas_correctas','latencia_mediana_s','coste_usd')])+' |')
    lines+=['','Las filas sin ejecución permanecen pendientes. Gemini por OpenRouter es una condición adicional;',
            'no sustituye la pareja Google base/ajustado.']
    if report['gemini_openrouter_demo']['casos']:lines+=['', 'Resultado adicional Gemini por OpenRouter: '+json.dumps(report['gemini_openrouter_demo'],ensure_ascii=False)]
    (output/'comparacion.md').write_text('\n'.join(lines)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
