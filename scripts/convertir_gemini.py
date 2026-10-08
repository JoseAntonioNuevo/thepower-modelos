import _bootstrap  # noqa: F401
import json
from curso.common import ROOT,read_jsonl,write_jsonl,assert_frozen,digest
from curso.remote import gemini_record


def main():
    assert_frozen();mapping={}
    for split in ("train","validacion","test"):
        rows=read_jsonl(ROOT/f"datos/{split}.jsonl")
        path=ROOT/f"datos/gemini/{split}.jsonl"
        write_jsonl(path,[gemini_record(row) for row in rows])
        mapping[split]={"ids_por_linea":[r["id"] for r in rows],"sha256":digest(path)}
    (ROOT/"datos/gemini/ids.json").write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+"\n")
    print("JSONL convertido; test separado de los archivos de entrenamiento y validación")


if __name__=="__main__":main()
