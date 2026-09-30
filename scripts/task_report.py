"""El informe de producto, y la pregunta que decide si el modelo ya sirve.

Genera el informe de riesgo por tarea DOS VECES sobre los mismos sujetos: una con
las etiquetas verdaderas y otra con lo que predice el modelo. El riesgo total es
idéntico en las dos —REBA se calcula de la geometría y no del modelo—, así que lo
único que puede cambiar es a qué tarea se le atribuye.

Y ahí está la pregunta que importa para el cliente, que no es el F1: **¿lleva el
informe del modelo a la misma decisión que el de la verdad?** Un modelo que se
equivoca a menudo pero señala las mismas tareas prioritarias ya sirve, porque el
jefe de planta interviene igual.

    .\\.venv\\Scripts\\python.exe scripts\\task_report.py --pliegue 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline.reba import reba_from_keypoints  # noqa: E402
from src.eval.harness import HZ, load  # noqa: E402
from src.eval.splits import cross_validation_folds  # noqa: E402
from src.models.train import train_fold  # noqa: E402
from src.product.report import build_report, ranking_agreement  # noqa: E402
from src.product.workstation import WorkstationConfig  # noqa: E402


def task_names(fields: dict[str, np.ndarray]) -> np.ndarray:
    """La tarea, tal como se nombra en el informe.

    Se deja fuera el objeto: para el riesgo importa qué se hace, con qué postura y
    a qué altura, no si es una caja o una varilla. Meterlo duplicaría cada fila del
    informe sin cambiar ninguna decisión.
    """
    return np.array(
        [
            f"{m} / {mn} / {h}"
            for m, mn, h in zip(fields["motion"], fields["manipulation"], fields["height"])
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Informe de riesgo por tarea.")
    parser.add_argument("--pliegue", type=int, default=1)
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--load-kg", type=float, default=None)
    parser.add_argument("--coupling", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    pliegue = [p for p in cross_validation_folds() if p.name == f"fold{args.pliegue}"][0]
    entrena = [load(i) for i in pliegue.train]
    prueba = [load(i) for i in pliegue.test]

    print(f"Entrenando en {list(pliegue.train)}, informando sobre {list(pliegue.test)}...")
    predicciones, _ = train_fold(entrena, prueba, args.epocas, args.device)

    config = WorkstationConfig(
        id="LINEA-3",
        name="Linea 3 - surtido de cajas",
        load_kg=args.load_kg,
        coupling=args.coupling,
    )

    # Se juntan los sujetos de prueba como si fueran turnos del mismo puesto: es lo
    # que vería el sistema instalado en una linea durante varios turnos.
    reba_todo, verdad_todo, predicho_todo, presente_todo = [], [], [], []
    for sujeto, prediccion in zip(prueba, predicciones):
        keypoints = sujeto.keypoints[sujeto.evaluable]
        datos = reba_from_keypoints(keypoints, assumptions=config.assumptions())
        reba_todo.append(datos)
        verdad_todo.append(
            task_names(
                {
                    "motion": sujeto.y("motion"),
                    "manipulation": sujeto.y("manipulation"),
                    "height": sujeto.y("height"),
                }
            )
        )
        predicho_todo.append(task_names(prediccion))
        presente_todo.append(np.ones(len(keypoints), dtype=bool))

    scores = {clave: np.concatenate([d[clave] for d in reba_todo]) for clave in reba_todo[0]}
    presente = np.concatenate(presente_todo)
    verdad = np.concatenate(verdad_todo)
    predicho = np.concatenate(predicho_todo)

    informe_verdad = build_report(verdad, scores, HZ, config, present=presente)
    informe_modelo = build_report(predicho, scores, HZ, config, present=presente)

    print("\n" + "=" * 78)
    print("INFORME CON LAS ETIQUETAS VERDADERAS (lo que veria un evaluador perfecto)")
    print("=" * 78)
    print(informe_verdad.format())

    print("\n" + "=" * 78)
    print("INFORME CON LO QUE PREDICE EL MODELO (lo que entrega el sistema)")
    print("=" * 78)
    print(informe_modelo.format())

    acuerdo = ranking_agreement(informe_verdad, informe_modelo)
    print("\n" + "=" * 78)
    print("LA PREGUNTA DE PRODUCTO: llevan los dos informes a la misma decision?")
    print("=" * 78)
    print(f"  la tarea numero 1 coincide:        {'SI' if acuerdo['top1_agree'] else 'NO'}")
    print(f"  de las {acuerdo['topn']} peores, coinciden:      {acuerdo['topn_overlap']}")
    print(
        f"  riesgo atribuido a la peor tarea:  {100 * acuerdo['share_truth']:.0f}% real "
        f"contra {100 * acuerdo['share_predicted']:.0f}% del modelo"
    )
    print(
        f"  minutos en riesgo (identicos, REBA no depende del modelo): "
        f"{acuerdo['risk_minutes_truth']:.1f}"
    )


if __name__ == "__main__":
    main()
