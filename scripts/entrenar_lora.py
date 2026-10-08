"""Configuración LoRA fija. No lee el test ni busca hiperparámetros."""
import _bootstrap  # noqa: F401
import argparse
import importlib.metadata
import json
import os
import time
import platform
from pathlib import Path
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
from curso.common import ROOT, assert_frozen, read_jsonl, basic_prompt


def training_output(path):
    output = Path(path).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Salida no vacía; usa otro --salida para conservar el entrenamiento anterior")
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--salida",default=str(ROOT/"modelos/lora-soporte"))
    args=parser.parse_args()
    output=training_output(args.salida)
    frozen=assert_frozen()
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTTrainer,SFTConfig
    from transformers import set_seed
    from curso.qwen import load_qwen,render
    set_seed(42)
    model,tokenizer,device=load_qwen()
    model.config.use_cache=False
    # Prompt/completion como texto ya templado, conserva el template idéntico a inferencia.
    def dataset(split):
        rows=[]
        for row in read_jsonl(ROOT/f"datos/{split}.jsonl"):
            msg=[{"role":"system","content":basic_prompt()}, {"role":"user","content":row["incidencia"]}]
            prompt=render(tokenizer,msg)
            whole=render(tokenizer,msg+[{"role":"assistant","content":json.dumps(row["objetivo"],ensure_ascii=False)}],False)
            if not whole.startswith(prompt): raise ValueError("Template entrenamiento/inferencia no coincide")
            # TRL agrega EOS a los textos que terminan en salto de línea.
            # Pretokenizar conserva exactamente el template, sin un EOS duplicado.
            input_ids=tokenizer(whole,add_special_tokens=False)["input_ids"]
            prompt_ids=tokenizer(prompt,add_special_tokens=False)["input_ids"]
            if input_ids[:len(prompt_ids)]!=prompt_ids:raise ValueError("Tokenización del prefijo distinta")
            if len(input_ids)>768:raise ValueError("Ejemplo excede max_length; no truncar respuesta silenciosamente")
            # TRL actual requiere labels explícitas al omitir su preparación.
            # -100 mantiene la pérdida solo sobre la respuesta, nunca el prompt.
            rows.append({"input_ids":input_ids,"attention_mask":[1]*len(input_ids),
                         "labels":[-100]*len(prompt_ids)+input_ids[len(prompt_ids):],
                         "completion_mask":[0]*len(prompt_ids)+[1]*(len(input_ids)-len(prompt_ids))})
        return Dataset.from_list(rows)
    output.mkdir(parents=True,exist_ok=True)
    settings={"r":8,"lora_alpha":16,"lora_dropout":0.05,
              "target_modules":["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
              "epochs":2,"learning_rate":0.0002,"batch_size":2,"gradient_accumulation_steps":4,
              "seed":42,"max_length":768}
    config=SFTConfig(output_dir=str(output/"checkpoints"),num_train_epochs=settings["epochs"],
        per_device_train_batch_size=2,per_device_eval_batch_size=2,gradient_accumulation_steps=4,
        learning_rate=0.0002,logging_steps=10,save_strategy="no",eval_strategy="epoch",
        max_length=768,packing=False,completion_only_loss=True,report_to="none",
        dataset_kwargs={"skip_prepare_dataset":True},
        fp16=False,bf16=False,seed=42,dataloader_pin_memory=False,optim="adamw_torch")
    trainer=SFTTrainer(model=model,processing_class=tokenizer,args=config,
        train_dataset=dataset("train"),eval_dataset=dataset("validacion"),
        peft_config=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",
                              target_modules=settings["target_modules"]))
    start=time.perf_counter()
    result=trainer.train()
    trainer.save_model(str(output));tokenizer.save_pretrained(str(output))
    receipt={"fecha":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"duracion_s":time.perf_counter()-start,
             "device":device,"hardware":f"{platform.system()} / {platform.machine()} / {device}",
             "configuracion":settings,"experimento":frozen,"metricas":result.metrics,
             "log_history":trainer.state.log_history,
             "versiones":{name:importlib.metadata.version(name) for name in ("torch","transformers","datasets","trl","peft","accelerate")},
             "coste_usd":None,"coste_metodo":"Entrenamiento local; energía/hardware no medidos"}
    (output/"entrenamiento.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"adapter":str(output),"duracion_s":receipt["duracion_s"],"device":device}))


if __name__=="__main__": main()
