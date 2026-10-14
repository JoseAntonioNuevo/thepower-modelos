# Gemini · repetir Nébula Hogar en tu propia cuenta

La [guía completa del laboratorio](obsidian/S8-Modelos-cerrados/05d-Gemini-manual-completo-fine-tuning.md)
explica qué hace SFT y todos los pasos: instalación, ADC, proyecto/bucket propios,
presupuesto, subida verificada, entrenamiento, checkpoints, inferencia,
comparación, resultados y límites, con diagramas Mermaid para Obsidian.

- [Dataset incluido](../datos/nebula-hogar/README.md): 328 train, 37 validación,
  78 test amplio, 48 test de precios y dos desarrollo; conversión y hashes.
- [CLI](../scripts/gemini_laboratorio.py) y [operaciones](../curso/gemini_laboratorio.py):
  tu proyecto explícito, sin recursos ni credenciales del profesor.
- [Referencia offline](referencias/nebula-hogar/README.md): 378 originales fechados,
  métricas y fallos del ensayo docente.

Desde la raíz del repositorio, con el entorno instalado:

```bash
python scripts/gemini_laboratorio.py comprobar
python scripts/gemini_laboratorio.py --help
```

No necesitan credenciales ni hacen inferencia. Las acciones Cloud requieren
tu configuración y, si escriben recursos o generan texto, `--autorizar-gasto`.
Preparar un JSON no entrena; una API key de AI Studio no reemplaza ADC de Cloud.
