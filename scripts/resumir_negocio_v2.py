"""Informe de tres condiciones sobre un test nuevo, sin reparar respuestas."""
import _bootstrap  # noqa: F401
import json
import math
from collections import Counter
from datetime import datetime,timezone
from curso.common import ROOT,read_jsonl
from curso.negocio_v2 import DATA,frozen,sha
from curso.negocio_v2_metricas import calibrated_rows,verify_registry
from gemini_negocio_v2 import save_new,summary


def main():
    frozen()
    registry=verify_registry()
    ev=ROOT/'material/evidencias/s8-gemini-negocio-v2-20261014'
    protocol=json.loads((DATA/'protocolo.json').read_text())
    selection=json.loads((ev/'seleccion-checkpoint.json').read_text())
    expected=[r['id'] for r in read_jsonl(DATA/'test.jsonl')]
    conditions=protocol['conditions']
    rows={condition:read_jsonl(ev/'test'/f'{condition}.jsonl') for condition in conditions}
    for condition,items in rows.items():
        if [r['id'] for r in items]!=expected:raise ValueError('Test incompleto o IDs distintos')
        if any(r['manifiesto_sha256']!=sha(DATA/'manifiesto.json') or r['split']!='test' or r['parametros_solicitados']!=protocol['generation'] for r in items if not r.get('error')):raise ValueError('Protocolo o generación distintos')
        if any(datetime.fromisoformat(r['fecha_utc'])<datetime.fromisoformat(selection['selected_at_utc']) for r in items):raise ValueError('Se abrió test antes de seleccionar checkpoint')
    legacy_totals={key:summary(items) for key,items in rows.items()}
    rows={key:calibrated_rows(items) for key,items in rows.items()}
    base=rows['base_sin_catalogo'];tuned=rows['ajustado_sin_catalogo']
    pair=Counter((b['valoracion']['caso_completo'],t['valoracion']['caso_completo']) for b,t in zip(base,tuned))
    wins=pair[False,True];losses=pair[True,False];discordant=wins+losses
    p_value=min(1.,2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/(2**discordant)) if discordant else 1.
    totals={key:summary(items) for key,items in rows.items()}
    gain=(totals['ajustado_sin_catalogo']['casos_completos']-totals['base_sin_catalogo']['casos_completos'])/len(expected)*100
    accuracy=totals['ajustado_sin_catalogo']['casos_completos']/len(expected)
    threshold=protocol['success_pre_registered']
    achieved=(accuracy>=threshold['min_complete_test_accuracy'] and gain>=threshold['min_gain_over_base_percentage_points'] and p_value<=threshold['paired_mcnemar_two_sided_p_max'])
    by_type={}
    for kind in sorted({r['tipo'] for r in base}):
        by_type[kind]={condition:{'n':sum(r['tipo']==kind for r in items),'correct':sum(r['tipo']==kind and r['valoracion']['caso_completo'] for r in items)} for condition,items in rows.items()}
    report={'evaluated_at_utc':datetime.now(timezone.utc).isoformat(),'summaries':totals,'by_type':by_type,'paired_counts':{'both_correct':pair[True,True],'both_incomplete':pair[False,False],'tuned_only_correct':wins,'base_only_correct':losses},'mcnemar_exact_two_sided_p':p_value,'tuned_complete_accuracy':accuracy,'gain_percentage_points':gain,'pre_registered_thresholds':threshold,'automatic_success_achieved':achieved,'review_note':'Automatic coverage metric; inspect content against catalog for contradictions and invented extra facts. No claim about a general-purpose benchmark.','selected_endpoint':selection['endpoint'],'source_test_sha256':sha(DATA/'test.jsonl'),'result_hashes':{key:sha(ev/'test'/f'{key}.jsonl') for key in conditions}}
    report['assessment_id']=registry['assessment_id']
    report['legacy_summaries']=legacy_totals
    report['calibration_registry']=registry
    save_new(ev/'comparacion.json',report)
    lines=['# Nébula Hogar v2 · test nuevo','',f"78 casos. Checkpoint elegido antes de test: `{selection['endpoint']}`.",'','| Condición | Casos completos | JSON | Esquema | Errores API |','|---|---:|---:|---:|---:|']
    for key,s in totals.items():lines.append(f"| {key} | {s['casos_completos']}/{s['casos']} | {s['json_valido']} | {s['esquema_valido']} | {s['errores_api']} |")
    lines+=['',f'Ventaja del ajustado sobre base sin catálogo: {gain:.2f} puntos. McNemar exacto bilateral: p={p_value:.8g}.',f"Criterio automático predefinido alcanzado: **{'sí' if achieved else 'no'}**.",'','La comparación principal mide conocimiento estático aprendido sin contexto de catálogo. El control recibe todos los datos como contexto. La cobertura automática no prueba veracidad total; la revisión de contenido se guarda aparte.','', '## Resultados por tipo','','| Tipo | Base | Ajustado | Base con catálogo |','|---|---:|---:|---:|']
    for kind,counts in by_type.items():lines.append('| '+kind+' | '+' | '.join(f"{counts[key]['correct']}/{counts[key]['n']}" for key in conditions)+' |')
    with (ev/'comparacion.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
