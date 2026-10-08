# Ejercicio · adaptar un modelo abierto

Ajusta Qwen3-0.6B para clasificar incidencias y compara instrucciones, few-shot
y LoRA con Transformers, Datasets, TRL y PEFT.

## Ejecución completa

1. Comprueba particiones, etiquetas y esquema del dataset sintético.
2. Registra modelo, revisión, dependencias y hardware.
3. Compara original, few-shot con train y adaptador LoRA.
4. Mismo template, thinking off y generación. No uses test para ajustar.
5. Conserva originales y métricas con un validador sobre los 30 casos.
6. Guarda el adaptador y verifica una predicción tras recargarlo.
7. Para completar publicación: cuenta propia de HF, repo nuevo privado si
   prefieres, model card y recarga por revisión. No exige endpoint contratado.

El dataset incluido usa 240/30/30. El alcance de preparación admite 150–300
train, 30 validación y 30 test. Una variante necesita sus propios datos,
manifiesto y resultados; no mezclarla con la referencia.

## Limitaciones del equipo

Se puede utilizar un entorno remoto con GPU si se dispone de acceso. También
se admite preparación completa cuando no hay ese acceso: datos comprobados,
código, configuración, comandos de entrenamiento/recarga, procedimiento de
evaluación y explicación de la limitación.

Marca métricas no obtenidas como pendientes. No inventes entrenamiento o
resultados. No es necesario contratar recursos de pago para esa modalidad.
Se valora comprensión, separación de datos, comparación y preparación; en la
modalidad ejecutada, evidencia del ajuste, recarga y análisis. No se exige que LoRA gane.
