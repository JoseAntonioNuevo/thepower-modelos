"""Genera un visor local de respuestas reales guardadas; no llama a modelos."""
import _bootstrap  # noqa: F401
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
from curso.common import ROOT,read_jsonl
from curso.negocio_v2 import DATA,frozen
from curso.negocio_v2_metricas import calibrated_rows
from gemini_negocio_v2 import summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['validacion','test'],default='test')
    p.add_argument('--caso',choices=['amplio','precios','precios-por-nombre'],default='amplio')
    args=p.parse_args()
    frozen()
    ev=ROOT/'material/evidencias/s8-gemini-negocio-v2-20261014'
    if args.caso!='amplio':
        paths=[ev/args.caso/f'{c}.jsonl' for c in ('base_sin_catalogo','ajustado_sin_catalogo','base_con_catalogo_control')]
        case_dir=ROOT/('experimentos/nebula-hogar-precios-nombre-v2' if args.caso=='precios-por-nombre' else 'experimentos/nebula-hogar-precios-v2')
        cases=read_jsonl(case_dir/'test.jsonl')
    elif args.split=='test':
        paths=[ev/'test'/f'{c}.jsonl' for c in ('base_sin_catalogo','ajustado_sin_catalogo','base_con_catalogo_control')]
        cases=read_jsonl(DATA/'test.jsonl')
    else:
        selected=json.loads((ev/'seleccion-checkpoint.json').read_text()) if (ev/'seleccion-checkpoint.json').exists() else None
        paths=[ev/'validacion/base.jsonl',ev/'validacion'/f"checkpoint-{selected['checkpoint_id']}.jsonl" if selected else None,ev/'validacion/base-con-catalogo.jsonl']
        cases=read_jsonl(DATA/'validacion.jsonl')
    labels=['Base sin catálogo','Ajustado sin catálogo','Base con catálogo · control']
    series=[]
    for label,path in zip(labels,paths):
        rows=calibrated_rows(read_jsonl(path)) if path and path.exists() else []
        series.append({'label':label,'summary':summary(rows) if rows else None,'records':{r['id']:r for r in rows}})
    snapshots=sorted((ev/'seguimiento').glob('*.json'))
    state=json.loads((ev/'estado-final.json').read_text()) if (ev/'estado-final.json').exists() else json.loads(snapshots[-1].read_text())
    data={'split':args.split if args.caso=='amplio' else args.caso,'generated_at_utc':datetime.now(timezone.utc).isoformat(),'job_state':state['state'],'cases':cases,'series':series}
    encoded=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    template=r'''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Nébula Hogar · resultados reales</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#10141d;color:#e9edf8;font:16px system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:30px 34px}header{display:flex;justify-content:space-between;align-items:start;gap:25px}h1{font-size:32px;margin:7px 0 12px}h2{font-size:20px;margin:0 0 14px}p{line-height:1.6;margin:8px 0}.eyebrow{color:#b29cff;font-size:13px;letter-spacing:.14em;text-transform:uppercase}.muted{color:#a9b4cc;font-size:13px}.tag{padding:8px 13px;border:1px solid #536282;border-radius:20px;font-size:13px;white-space:nowrap}.metrics,.answers{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.metrics{margin:22px 0}.card{background:#192131;border:1px solid #38455e;border-radius:14px;padding:20px}.card:nth-child(2){border-color:#af92ff}.big{font-size:36px;color:#e2d7ff;margin:8px 0}.question{background:#1b3440;border-left:4px solid #44cbd7;border-radius:10px;padding:18px;margin:18px 0}.controls{display:flex;flex-wrap:wrap;gap:8px}button{cursor:pointer;background:#20293d;border:1px solid #526281;border-radius:8px;color:#e8efff;padding:9px 14px;font:inherit;font-size:13px}button[aria-pressed=true]{background:#6b4faa;border-color:#b293ff}input{margin:14px 0 2px;display:block;width:100%;padding:11px;background:#111927;border:1px solid #536282;color:#f4f6ff;border-radius:8px}details{margin-top:14px;font-size:13px}summary{cursor:pointer;color:#a8dce5}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;color:#cad6ed;line-height:1.45}.verdict{font-size:13px;color:#69ddba;margin-bottom:12px}.bad{color:#f1b78a}.answer{font-size:17px;min-height:105px;line-height:1.6}.ids{font-size:12px;color:#b4c0db;margin-top:12px}.source{margin-top:16px;border:1px solid #334863;border-radius:10px;padding:15px;color:#bfd4ee;font-size:14px}footer{margin-top:20px;color:#97a5be;font-size:12px}@media(max-width:850px){.metrics,.answers{grid-template-columns:1fr}header{display:block}main{padding:20px}}
</style><main><header><div><div class="eyebrow">S8 · Gemini aprende un negocio ficticio</div><h1>Nébula Hogar: comparación real</h1><p>Las mismas preguntas, tres condiciones. Respuestas del ensayo guardadas, sin consultas al navegar.</p></div><div class="tag" id="stage"></div></header><div class="metrics" id="metrics"></div><div class="controls" id="buttons"></div><input id="query" placeholder="Buscar entre todas las preguntas por texto o ID" aria-label="Buscar preguntas"><div class="controls" id="matches"></div><section class="question"><div class="muted" id="caseid"></div><h2 id="question"></h2></section><div class="answers" id="answers"></div><div class="source" id="source"></div><footer id="footer"></footer></main><script type="application/json" id="data">__DATA__</script><script>
const d=JSON.parse(document.getElementById('data').textContent);const $=id=>document.getElementById(id);let selected=d.cases[0].id;
$('stage').textContent=d.split.toUpperCase()+' · '+d.cases.length+' casos';
d.series.forEach(s=>{const el=document.createElement('div');el.className='card';const h=document.createElement('h2');h.textContent=s.label;el.append(h);const n=document.createElement('div');n.className='big';n.textContent=s.summary?s.summary.casos_completos+'/'+s.summary.casos:'Pendiente';el.append(n);const p=document.createElement('p');p.className='muted';p.textContent=s.summary?'Casos completos según el protocolo · '+s.summary.errores_api+' errores de API':'Todavía no hay resultados de esta condición';el.append(p);$('metrics').append(el)});
const examples=d.split==='precios-por-nombre'?['NH-NOM-001','NH-NOM-006','NH-NOM-011','NH-NOM-016','NH-NOM-021','NH-NOM-026','NH-NOM-031','NH-NOM-036','NH-NOM-041']:d.split==='precios'?['NH-PRE-001','NH-PRE-002','NH-PRE-013','NH-PRE-049']:d.split==='test'?['NH-TE-041','NH-TE-044','NH-TE-049','NH-TE-069','NH-TE-073']:['NH-VA-017','NH-VA-020','NH-VA-025','NH-VA-033'];const names=d.split==='precios-por-nombre'?['Órbita','Alba','Faro','Marea','Nido','Duna','Prisma','Línea','Fuera del catálogo']:d.split==='precios'?['Precio por nombre','Error por SKU','Termo Faro','Fuera del catálogo']:['Café caliente','Carga Qi','Compra con envío','Pedido o stock real','Límite del asistente'];
function button(c,text,parent){const b=document.createElement('button');b.textContent=text;b.dataset.caseid=c.id;b.onclick=()=>show(c.id);parent.append(b)}
examples.forEach((id,i)=>{const c=d.cases.find(x=>x.id===id);if(c)button(c,names[i],$('buttons'))});
function show(id){selected=id;const c=d.cases.find(x=>x.id===id);$('caseid').textContent=c.id+' · '+c.tipo;$('question').textContent=c.pregunta;$('answers').replaceChildren();d.series.forEach(s=>{const card=document.createElement('div');card.className='card';const h=document.createElement('h2');h.textContent=s.label;card.append(h);const r=s.records[id];const status=document.createElement('div');status.className='verdict'+(r&&!r.valoracion.caso_completo?' bad':'');status.textContent=r?(r.valoracion.caso_completo?'Caso completo':'Caso incompleto · conservar el original'):'Pendiente';card.append(status);let obj=null;try{obj=JSON.parse(r.respuesta_original)}catch(e){}const p=document.createElement('div');p.className='answer';p.textContent=obj&&typeof obj.respuesta==='string'?obj.respuesta:r?(r.respuesta_original||'Error: '+r.error):'El modelo todavía no se ha evaluado en esta condición.';card.append(p);const ids=document.createElement('div');ids.className='ids';ids.textContent=obj?'Estado: '+obj.estado+' · Productos: '+JSON.stringify(obj.productos):'';card.append(ids);if(r){const detail=document.createElement('details');const summary=document.createElement('summary');summary.textContent='Ver JSON original y recurso';detail.append(summary);const pre=document.createElement('pre');pre.textContent=r.respuesta_original+'\n\nModelo: '+r.model+'\nFecha UTC: '+r.fecha_utc;detail.append(pre);card.append(detail)}$('answers').append(card)});$('source').textContent='Respuesta objetivo escrita antes de entrenar: '+c.objetivo.respuesta;document.querySelectorAll('button[data-caseid]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.caseid===id));}
$('query').oninput=()=>{const q=$('query').value.toLocaleLowerCase();$('matches').replaceChildren();if(!q)return;d.cases.filter(c=>(c.id+' '+c.pregunta).toLocaleLowerCase().includes(q)).slice(0,10).forEach(c=>button(c,c.id+' · '+c.pregunta.slice(0,72),$('matches')))};
$('footer').textContent='Captura del informe local: '+d.generated_at_utc+' · Estado registrado de Google: '+d.job_state+' · Catálogo sintético versionado; sin acceso a pedidos o inventario en directo. La cobertura automática exige revisión contra fuente.';show(examples[0]||selected);
</script></html>'''
    out=ev/'visor';out.mkdir(exist_ok=True)
    path=out/f'{args.split if args.caso=="amplio" else args.caso}.html'
    path.write_text(template.replace('__DATA__',encoded))
    print(path)


if __name__=='__main__':main()
