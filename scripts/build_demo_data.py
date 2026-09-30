"""Genera los datos del panel: cuatro puestos con sus turnos, informes y replays.

Entrena un modelo por pliegue y **predice cada turno con el modelo que NO lo vio
entrenando**. La demo enseña predicciones sobre gente que el modelo no conoce, que
es lo que le pasaría instalado en una planta; enseñar predicciones sobre los datos
de entrenamiento daría una demo más lucida y sería mentir.

    .\\.venv\\Scripts\\python.exe scripts\\build_demo_data.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.harness import HZ, load  # noqa: E402
from src.eval.splits import cross_validation_folds  # noqa: E402
from src.models.train import train_fold  # noqa: E402
from src.product.pipeline import DEMO_WORKSTATIONS, build_workstation, process_shift  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "artifacts/demo/panel.json"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pliegues = cross_validation_folds()
    print(f"Generando el panel con {len(pliegues)} pliegues en {device}.")

    puestos = []
    for config, pliegue in zip(DEMO_WORKSTATIONS, pliegues):
        entrena = [load(i) for i in pliegue.train]
        prueba = [load(i) for i in pliegue.test]
        print(f"  {config.id}: entrena con {list(pliegue.train)}, turnos {list(pliegue.test)}")

        predicciones, _ = train_fold(entrena, prueba, epochs=30, device=device)
        turnos = [
            process_shift(sujeto, prediccion, config)
            for sujeto, prediccion in zip(prueba, predicciones)
        ]
        puesto = build_workstation(config, turnos)
        puestos.append(puesto)
        print(
            f"      {puesto['seconds'] / 60:.1f} min, "
            f"{puesto['seconds_at_risk'] / 60:.1f} min en riesgo, "
            f"{len(puesto['events'])} eventos"
        )

    puestos.sort(key=lambda p: -p["seconds_at_risk"])
    salida = {
        "generado": date.today().isoformat(),
        "hz": HZ,
        "fuente": "UW-IOM, cada participante es un turno. Predicciones fuera de pliegue.",
        "puestos": puestos,
    }

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_text(json.dumps(salida, ensure_ascii=False), encoding="utf-8")
    print(f"\n{DESTINO.relative_to(RAIZ)}  ({DESTINO.stat().st_size / 1e6:.1f} MB)")
    print("ranking de puestos por exposicion:")
    for p in puestos:
        marca = "" if p["configured"] else "  (sin configurar: cota inferior)"
        print(f"  {p['id']:<12}{p['seconds_at_risk'] / 60:>7.1f} min en riesgo{marca}")


if __name__ == "__main__":
    main()
