# Rutas opcionales · Gemini, Claude y Hugging Face

La práctica local no necesita claves externas. Estas rutas requieren cuentas
propias y pueden generar costes; no vienen recursos cloud creados.

## OpenRouter

Configura OPENROUTER_API_KEY en tu entorno/.env local. Verifica disponibilidad
y precios de las referencias del curso con tu cuenta antes de llamar.

```bash
python scripts/demo.py --proveedor openrouter \
  --modelo google/gemini-3.1-flash-lite --caso DE-04
python scripts/demo.py --proveedor openrouter \
  --modelo anthropic/claude-haiku-4.5 --caso DE-04 --fewshot
```

Gemini/OpenRouter no sustituye el par base/ajustado de Google Cloud.

## Preparación de Gemini sin crear recursos

```bash
python scripts/convertir_gemini.py
python scripts/gemini_tuning.py preparar \
  --proyecto ID_PROYECTO_CURSO --bucket BUCKET_CURSO
```

Los nombres son marcadores. `preparar` es offline. Train/validación se convierten
a contents/parts; test queda separado. `ids.json` mantiene correspondencia.

## Ajuste real de Gemini

Necesitas proyecto identificado, permisos, facturación/presupuesto, bucket,
cuota y acceso al modelo. Configura .env.example y autentica tu cuenta con
Google. No tomar un proyecto personal por defecto como laboratorio.

La receta usa europe-west4 para tuning y eu para serving. Verifica soporte,
regiones y tarifas actuales. Estima tokens, épocas, almacenamiento e inferencia.
El presupuesto del script es una comprobación previa, no un tope de Google.

Solo con recursos/gasto preparados usa `subir-datos` y `crear`, indicando
`--proyecto`, `--bucket`, `--presupuesto-usd` y `--coste-estimado-usd`.
Conserva el ID. `estado --proyecto ... --job ...` consulta ese job; `crear` de
nuevo inicia otro. La comparación base/ajustado mantiene servicio, datos,
instrucciones y thinking MINIMAL. Evidencia: job terminado, modelo y predicciones.

## Publicar el adaptador Qwen

Revisa archivos y utiliza un nombre nuevo en tu cuenta:

```bash
python scripts/publicar_adaptador.py \
  --repo TU_USUARIO/NOMBRE_NUEVO --adaptador modelos/lora-soporte
```

Verifica el usuario, crea un repo privado nuevo, sube pesos/configuración/model
card/receta y prueba recarga por commit. Configura la credencial fuera del repo.
No compartir claves ni cachés.
