"""Descargar únicamente el checkpoint público fijado para la práctica local."""
import _bootstrap  # noqa: F401
import json

from curso.common import MODEL_ID, REVISION, assert_frozen


def main():
    assert_frozen()
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID, revision=REVISION, token=False, max_workers=2,
        allow_patterns=[
            "config.json", "generation_config.json", "*.safetensors", "*.safetensors.index.json",
            "tokenizer*", "vocab.json", "merges.txt", "added_tokens.json",
            "special_tokens_map.json", "chat_template.jinja",
        ],
    )
    print(json.dumps({"modelo": MODEL_ID, "revision": REVISION, "descargado": True}))


if __name__ == "__main__":
    main()
