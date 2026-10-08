"""Inferencia y tokenización con el mismo template, sin llamadas al importar."""
import json
import os
import time
from pathlib import Path
from .common import MODEL_ID, REVISION, GENERATION, messages


def load_qwen(adapter=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # FP32 en MPS facilita entrenamiento reproducible de este modelo pequeño.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
               token=False, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION,
               dtype=torch.float32 if device=="mps" else "auto", attn_implementation="eager",
               token=False, trust_remote_code=False, use_safetensors=True).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer, device


def render(tokenizer, msg, add_generation_prompt=True):
    return tokenizer.apply_chat_template(msg, tokenize=False,
           add_generation_prompt=add_generation_prompt, enable_thinking=False)


def generate(model, tokenizer, device, incident, fewshot=False):
    import torch
    text = render(tokenizer,messages(incident,fewshot))
    inputs = tokenizer(text,return_tensors="pt",add_special_tokens=False).to(device)
    if device=="mps": torch.mps.synchronize()
    start=time.perf_counter()
    with torch.inference_mode():
        output=model.generate(**inputs,**GENERATION, pad_token_id=tokenizer.eos_token_id)
    if device=="mps": torch.mps.synchronize()
    tokens=output[0,inputs["input_ids"].shape[1]:]
    return {"respuesta_original": tokenizer.decode(tokens,skip_special_tokens=True),
            "latencia_s": time.perf_counter()-start,
            "tokens_entrada": inputs["input_ids"].shape[1], "tokens_salida": len(tokens),
            "coste_usd": None, "coste_metodo": "Sin tarifa API; hardware/energía no medidos",
            "hardware":device, "model":MODEL_ID, "revision":REVISION,
            "parametros":GENERATION, "enable_thinking":False}
