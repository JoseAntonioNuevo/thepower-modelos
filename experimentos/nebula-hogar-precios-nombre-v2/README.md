# Caso final de la clase · precios por nombre

Este caso reutiliza el entrenamiento real de `../nebula-hogar-v2/`: 328 train,
37 validación, ocho épocas de Gemini 3.1 Flash-Lite y checkpoint final seleccionado
con validación. No se entrenó otro modelo ni se alteró el test amplio.

El modelo aprendió referencias y precios, pero el intento amplio conserva fallos
de composición, colores, carritos y consultas por SKU aislado. La aplicación del
directo se limita a preguntas de PVP por el **nombre completo** del producto y a
rechazar artículos ajenos al catálogo. No cubre inventario vivo ni acciones reales.

Se registró un **test nuevo de 48 preguntas** después de desarrollar el caso:
40 consultas de precio, cinco por cada uno de ocho productos, y ocho nombres
ajenos nuevos. Resultado: base sin catálogo 8/48, ajustado sin catálogo 48/48,
base con catálogo 48/48. Son ocho precios independientes y variantes de preguntas,
no 48 productos ni una garantía de rendimiento para otros negocios.

Los resultados y capturas son locales del profesor; los archivos aquí contienen
preguntas, respuestas esperadas y el registro previo con hashes.

Desde la raíz del repositorio:

```bash
.venv/bin/python scripts/consultar_negocio_v2.py \
  --pregunta "¿Cuánto cuesta el Cuaderno Prisma de Nébula Hogar?"
```

El script consulta el endpoint real elegido, usa ADC del laboratorio y consume
inferencia. El precio correcto de esta versión es 12 € con IVA. El comando con
`--base` permite la comparación con el mismo identificador de modelo base.
