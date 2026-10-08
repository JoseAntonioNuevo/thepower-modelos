import _bootstrap  # noqa: F401
import argparse,importlib.util,json,os,platform
from curso.common import ROOT,validate_splits,assert_frozen
from curso.remote import load_config
from curso.http import bounded_json,model_ids


def main():
    p=argparse.ArgumentParser();p.add_argument('--local',action='store_true');args=p.parse_args();load_config()
    result={'python':platform.python_version(),'splits':validate_splits(),'experimento_congelado':bool(assert_frozen()),
            'dependencias':{m:importlib.util.find_spec(m) is not None for m in ('requests','transformers','torch','trl','peft','datasets','google.genai')},
            'openrouter_configurado':bool(os.getenv('OPENROUTER_API_KEY')),
            'google_proyecto_curso_configurado':bool(os.getenv('GOOGLE_CLOUD_PROJECT')),
            'adaptador_local':(ROOT/'modelos/lora-soporte/adapter_model.safetensors').exists(),
            'datos_sinteticos':True,'conexion_remota':'No probada; preflight no consume inferencia'}
    if args.local:
        import requests
        url=os.getenv('LOCAL_BASE_URL','http://localhost:1234/v1').rstrip('/')+'/models'
        try:
            response=requests.get(url,timeout=5,stream=True,allow_redirects=False)
            try:
                if response.status_code!=200:raise ValueError("Catálogo HTTP no disponible")
                response.raise_for_status()
                result['modelos_locales']=model_ids(bounded_json(response,max_seconds=10))
            finally:response.close()
        except (requests.RequestException,ValueError):result['modelos_locales']='Servidor local no disponible o respuesta inválida'
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
