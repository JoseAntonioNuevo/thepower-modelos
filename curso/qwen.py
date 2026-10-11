# GUÍA DE LECTURA · curso/qwen.py
# Puente entre los archivos del Hub y la inferencia local con Transformers.
# load_qwen carga la misma revisión para todas las condiciones.
# render aplica la plantilla de conversación y generate produce tokens nuevos.
# El adaptador LoRA se carga sobre la base; la inferencia no actualiza parámetros.
# Las dependencias pesadas se importan dentro de las funciones que las necesitan.

"""Inferencia y tokenización con el mismo template, sin llamadas al importar."""

# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
import json

# Lee configuración del entorno y ajusta opciones del proceso.
import os

# Mide duración y prepara fechas; perf_counter/monotonic sirven para intervalos, no fechas del calendario.
import time

# Path permite construir rutas, comprobar archivos y leer/escribir sin concatenar separadores a mano.
from pathlib import Path
from .common import MODEL_ID, REVISION, GENERATION, messages


# Carga tokenizer y modelo base en el dispositivo disponible.
# Si adapter tiene una ruta, aplica ese adaptador PEFT sobre la misma base.
# Devuelve modelo, tokenizer y nombre del dispositivo; todavía no genera texto.
def load_qwen(adapter=None):
    # Motor de tensores y gradientes; durante inferencia se desactivan los gradientes.
    import torch

    # Carga modelos/tokenizers y aporta generación y utilidades de entrenamiento.
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Prefiere CUDA; si no existe, MPS del Mac; como última opción utiliza CPU.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # FP32 en MPS facilita entrenamiento reproducible de este modelo pequeño.
    # Carga vocabulario y plantilla de la misma revisión del modelo, sin ejecutar código del repositorio.
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, token=False, trust_remote_code=False
    )
    # Carga los pesos Safetensors: FP32 en MPS, dtype del checkpoint en otros dispositivos, y los mueve al equipo.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        dtype=torch.float32 if device == "mps" else "auto",
        attn_implementation="eager",
        token=False,
        trust_remote_code=False,
        use_safetensors=True,
    ).to(device)
    # Solo aplica PEFT si se ha solicitado un adaptador; la base sola es la condición original.
    if adapter:
        # Añade/recarga adaptadores LoRA manteniendo el modelo base separado.
        from peft import PeftModel

        # Recarga las matrices aprendidas sobre el modelo original compatible.
        model = PeftModel.from_pretrained(model, adapter)
    # Activa modo evaluación para desactivar dropout; SFTTrainer activará modo entrenamiento cuando corresponda.
    model.eval()
    return model, tokenizer, device


# Convierte una lista de mensajes en texto con la plantilla oficial de Qwen.
# add_generation_prompt prepara el comienzo de la respuesta; thinking queda desactivado.
def render(tokenizer, msg, add_generation_prompt=True):
    # Marca roles y turno assistant con el template de Qwen; enable_thinking=False conserva el protocolo.
    return tokenizer.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


# Genera una respuesta local usando el prompt compartido y parámetros fijados.
# Recibe incidencia y fewshot, y devuelve texto original, tokens y tiempo.
# Durante inferencia no calcula gradientes ni actualiza los pesos.
def generate(model, tokenizer, device, incident, fewshot=False):
    # Motor de tensores y gradientes; durante inferencia se desactivan los gradientes.
    import torch

    # Texto formateado o decodificado que se pasa al siguiente paso del recorrido.
    text = render(tokenizer, messages(incident, fewshot))
    # Convierte el prompt en tensores y los mueve al mismo dispositivo que los pesos.
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    # Espera a las operaciones MPS pendientes para que la medida temporal tenga límites definidos.
    if device == "mps":
        torch.mps.synchronize()
    # Inicio de la medida temporal; el tiempo de carga de pesos se mide por separado cuando corresponde.
    start = time.perf_counter()
    # Desactiva el seguimiento de gradientes: consultar el modelo no lo entrena.
    with torch.inference_mode():
        # Predice tokens nuevos con generación greedy y límite de 180 tokens fijados en GENERATION.
        output = model.generate(
            **inputs, **GENERATION, pad_token_id=tokenizer.eos_token_id
        )
    # Espera a las operaciones MPS pendientes para que la medida temporal tenga límites definidos.
    if device == "mps":
        torch.mps.synchronize()
    # Separa la respuesta generada de los tokens de entrada que también aparecen en output.
    tokens = output[0, inputs["input_ids"].shape[1] :]
    # Decodifica la salida y devuelve consumo/tiempo; coste local sigue desconocido porque no medimos energía/hardware.
    return {
        "respuesta_original": tokenizer.decode(tokens, skip_special_tokens=True),
        "latencia_s": time.perf_counter() - start,
        "tokens_entrada": inputs["input_ids"].shape[1],
        "tokens_salida": len(tokens),
        "coste_usd": None,
        "coste_metodo": "Sin tarifa API; hardware/energía no medidos",
        "hardware": device,
        "model": MODEL_ID,
        "revision": REVISION,
        "parametros": GENERATION,
        "enable_thinking": False,
    }
