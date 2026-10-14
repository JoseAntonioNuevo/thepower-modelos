"""Laboratorio reproducible de Nébula Hogar con la cuenta Cloud del alumno.

Ejecuta --help. Preparar/comprobar no necesitan credenciales; las acciones de
subida, creación o inferencia requieren --autorizar-gasto. Los recursos se leen
de tu laboratorio, nunca de una cuenta del profesor o del proyecto global ADC.
"""

import _bootstrap  # noqa: F401
import argparse
import json
import sys
import time
from pathlib import Path

from curso import gemini_laboratorio as lab
from curso.negocio import CONDITIONS, dataset_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accion", choices=("comprobar", "preparar", "contar-tokens", "subir-datos", "crear",
                                          "estado", "vincular", "checkpoints", "seleccionar", "evaluar", "consultar", "resumir"))
    parser.add_argument("--laboratorio", type=Path, default=Path("resultados/gemini-nebula"))
    parser.add_argument("--proyecto", help="Tu ID de proyecto; obligatorio al preparar")
    parser.add_argument("--bucket", help="Tu bucket privado, nombre simple; obligatorio al preparar")
    parser.add_argument("--nombre", help="Prefijo/nombre único para tu ensayo; obligatorio al preparar")
    parser.add_argument("--presupuesto-usd", type=float)
    parser.add_argument("--epocas", type=int, default=8)
    parser.add_argument("--region", choices=tuple(lab.REGIONS), default="europe-west4")
    parser.add_argument("--tarifa-entrenamiento-usd-millon", type=float, default=3.0)
    parser.add_argument("--reserva-inferencia-usd", type=float, default=0.5)
    parser.add_argument("--autorizar-gasto", action="store_true", help="Autoriza solo esta acción en tu laboratorio")
    parser.add_argument("--job", help="Recurso existente: solo para reconciliar una creación incierta con vincular")
    parser.add_argument("--condicion", choices=CONDITIONS)
    parser.add_argument("--split", choices=("validacion", "test", "test-precios"), default="validacion")
    parser.add_argument("--checkpoint", help="ID de un candidato propio para evaluar validación")
    parser.add_argument("--pregunta")
    parser.add_argument("--esperar", action="store_true", help="Solo estado: seguir el mismo job hasta fin o 120 minutos")
    args = parser.parse_args()
    try:
        if args.accion == "comprobar":
            result = {"estado": "OFFLINE OK", "dataset": dataset_check()}
        elif args.accion == "preparar":
            if not all((args.proyecto, args.bucket, args.nombre, args.presupuesto_usd)):
                parser.error("preparar requiere --proyecto, --bucket, --nombre y --presupuesto-usd")
            result = lab.prepare(args.laboratorio, args.proyecto, args.bucket, args.nombre,
                                 args.presupuesto_usd, args.epocas, args.region,
                                 args.tarifa_entrenamiento_usd_millon, args.reserva_inferencia_usd)
        else:
            cfg = lab.load_run(args.laboratorio)
            if args.accion == "contar-tokens":
                result = lab.estimate(args.laboratorio, cfg)
            elif args.accion == "subir-datos":
                result = lab.upload(args.laboratorio, cfg, args.autorizar_gasto)
            elif args.accion == "crear":
                result = lab.create(args.laboratorio, cfg, args.autorizar_gasto)
            elif args.accion in ("estado", "vincular"):
                if args.accion == "vincular" and not args.job:
                    parser.error("vincular requiere --job del mismo proyecto y dataset")
                result = lab.status(args.laboratorio, cfg, args.job if args.accion == "vincular" else None)
                if args.esperar and args.accion == "estado":
                    started = time.monotonic()
                    previous = None
                    while result["state"] not in lab.TERMINAL:
                        if result["state"] != previous:
                            print(result["state"], file=sys.stderr, flush=True)
                            previous = result["state"]
                        if time.monotonic() - started >= 120 * 60:
                            parser.exit(2, "Espera local agotada; el job continúa en Google. Consulta estado otra vez.\n")
                        time.sleep(45)
                        result = lab.status(args.laboratorio, cfg)
            elif args.accion == "checkpoints":
                result = lab.candidates(args.laboratorio, cfg)
            elif args.accion == "seleccionar":
                result = lab.choose(args.laboratorio, cfg)
            elif args.accion == "evaluar":
                if not args.condicion:
                    parser.error("evaluar requiere --condicion")
                result = lab.evaluate(args.laboratorio, cfg, args.condicion, args.split,
                                      args.checkpoint, args.autorizar_gasto)
            elif args.accion == "consultar":
                if not args.condicion or not args.pregunta:
                    parser.error("consultar requiere --condicion y --pregunta")
                result = lab.query(args.laboratorio, cfg, args.condicion, args.pregunta, args.autorizar_gasto)
            else:
                result = lab.compare(args.laboratorio, cfg, args.split)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
