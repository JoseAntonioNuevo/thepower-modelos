# Protocolo del experimento

Clasificar incidencias ficticias en acceso, facturacion, tecnico o consulta;
asignar alta/media/baja y redactar una o dos frases. Reglas en
`prompts/basico.txt`.

## Datos

Train: 240 registros, 60 por categoría, a partir de 40 situaciones iniciales
con seis variantes de prefijo. Validación y test: 30 cada uno. Desarrollo: 6.
Se comprueban IDs/textos/familias separados. La diversidad sintética es limitada.

`id` y `familia` son metadatos. Entrenamiento utiliza instrucciones, incidencia
y respuesta objetivo. Template común, thinking off y tokens sin duplicación.
Solo la completion aporta pérdida directa.

Validación se evalúa por época sin elegir automáticamente un checkpoint.
`config/experimento-congelado.json` registra hashes y revisión. No regenerar
ni ajustar después de leer test. Una variante requiere su propio experimento.

## Condiciones

1. Qwen original con instrucciones básicas.
2. El mismo Qwen con cuatro ejemplos de train en cada consulta.
3. El mismo Qwen con LoRA y las instrucciones básicas.

LoRA: dos épocas, r=8, alpha=16, dropout=0.05 y configuración fija.
Comparar con Qwen3.5-4B cambiaría también el modelo.

## Métricas

JSON, esquema, categoría, prioridad y ambas etiquetas se miden por separado.
Una salida no interpretable y los fallos de API siguen en el denominador.
Originales conservados, sin JSON mode, tools que impongan formato, reparación
ni reintentos correctores.

La utilidad del texto necesita revisión humana. Los objetivos genéricos pueden
producir etiquetas correctas y respuestas poco útiles.
La API web serializa peticiones: su tiempo puede incluir cola. Coste desconocido
se conserva como null. Entrenamiento e inferencia se contabilizan por separado.

`resultados-referencia.json` contiene cifras de una ejecución anterior fechada.
Las nuevas se guardan por separado. LoRA no tiene que ganar; 30 casos no
establecen un ganador universal.
