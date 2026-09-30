"""Entrena el TCN y lo compara con la línea base, sobre los MISMOS fotogramas.

Los dos brazos pasan por el mismo arnés, así que la diferencia entre sus números es
la diferencia entre los métodos y nada más. El reservado no se toca: esto son los
cuatro pliegues de desarrollo.

    .\\.venv\\Scripts\\python.exe scripts\\train_model.py
    .\\.venv\\Scripts\\python.exe scripts\\train_model.py --epocas 40 --solo-pliegue 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline.rules import calibrate, predict_all  # noqa: E402
from src.datasets.uwiom import LABEL_FIELDS  # noqa: E402
from src.eval.harness import HZ, evaluate, load  # noqa: E402
from src.eval.metrics import majority_prediction, report  # noqa: E402
from src.eval.splits import cross_validation_folds  # noqa: E402
from src.models.tcn import SkeletonTCN, class_weights, normalize_windows  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
SEMILLA = 20260929


def entrenar_pliegue(entrena, prueba, epocas: int, device: str, verbose: bool):
    """Entrena el TCN en los sujetos de entrenamiento y predice sobre los de prueba."""
    torch.manual_seed(SEMILLA)
    np.random.seed(SEMILLA)

    x_train = normalize_windows(np.concatenate([s.windows() for s in entrena]))
    y_train = {c: np.concatenate([s.y(c) for s in entrena]) for c in LABEL_FIELDS}
    clases = {c: sorted(set(y_train[c].tolist())) for c in LABEL_FIELDS}
    indices = {c: {v: i for i, v in enumerate(clases[c])} for c in LABEL_FIELDS}

    xt = torch.tensor(x_train, device=device)
    yt = {
        c: torch.tensor([indices[c][v] for v in y_train[c]], device=device)
        for c in LABEL_FIELDS
    }

    modelo = SkeletonTCN({c: len(clases[c]) for c in LABEL_FIELDS}).to(device)
    perdidas = {
        c: nn.CrossEntropyLoss(weight=class_weights(y_train[c], clases[c]).to(device))
        for c in LABEL_FIELDS
    }
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=2e-3, weight_decay=1e-4)
    planificador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=epocas)

    n, lote = len(xt), 256
    modelo.train()
    for epoca in range(epocas):
        orden = torch.randperm(n, device=device)
        total = 0.0
        for inicio in range(0, n, lote):
            trozo = orden[inicio : inicio + lote]
            salida = modelo(xt[trozo])
            perdida = sum(perdidas[c](salida[c], yt[c][trozo]) for c in LABEL_FIELDS)
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()
            total += float(perdida.detach()) * len(trozo)
        planificador.step()
        if verbose and (epoca + 1) % 10 == 0:
            print(f"      epoca {epoca + 1:>3}  perdida {total / n:.4f}")

    modelo.eval()
    predicciones = []
    with torch.no_grad():
        for sujeto in prueba:
            x = torch.tensor(normalize_windows(sujeto.windows()), device=device)
            salida = modelo(x)
            predicciones.append(
                {
                    c: np.array([clases[c][i] for i in salida[c].argmax(dim=1).cpu().numpy()])
                    for c in LABEL_FIELDS
                }
            )
    return predicciones, modelo.n_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el TCN y compáralo con la línea base.")
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--solo-pliegue", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cache: dict[int, object] = {}

    def datos(i):
        if i not in cache:
            cache[i] = load(i)
        return cache[i]

    pliegues = cross_validation_folds()
    if args.solo_pliegue:
        pliegues = [p for p in pliegues if p.name == f"fold{args.solo_pliegue}"]

    print(f"TCN causal, {args.epocas} epocas, {args.device}. Los dos brazos, mismos fotogramas.\n")
    resultados: dict[str, dict[str, list]] = {
        nombre: {"degenerado": [], "reglas": [], "tcn": []}
        for nombre in (*LABEL_FIELDS, "etiqueta completa")
    }
    parametros = 0

    for pliegue in pliegues:
        entrena = [datos(i) for i in pliegue.train]
        prueba = [datos(i) for i in pliegue.test]
        evaluables = sum(len(s.evaluable) for s in prueba)
        print(f"  {pliegue.name}: prueba {list(pliegue.test)}, {evaluables} fotogramas evaluables")

        # --- brazo 1: el degenerado ---
        y_train_campos = {
            c: np.concatenate([s.y(c) for s in entrena]) for c in LABEL_FIELDS
        }
        deg = [
            {c: majority_prediction(y_train_campos[c], len(s.evaluable)) for c in LABEL_FIELDS}
            for s in prueba
        ]

        # --- brazo 2: las reglas, calibradas solo en entrenamiento ---
        muestras = [(s.keypoints, s.trunk_flexion, s.labels) for s in entrena]
        umbrales = calibrate(muestras, HZ)
        reglas = []
        for s in prueba:
            completo = predict_all(s.keypoints, s.trunk_flexion, HZ, umbrales)
            reglas.append({c: completo[c][s.evaluable] for c in LABEL_FIELDS})

        # --- brazo 3: el TCN ---
        comenzado = time.perf_counter()
        tcn, parametros = entrenar_pliegue(entrena, prueba, args.epocas, args.device, args.verbose)
        print(f"      entrenado en {time.perf_counter() - comenzado:.0f} s")

        for nombre, brazo in (("degenerado", deg), ("reglas", reglas), ("tcn", tcn)):
            for campo, informe in evaluate(prueba, brazo).items():
                resultados[campo][nombre].append(informe)

    print(f"\nTCN: {parametros:,} parametros\n")
    print(f"{'campo':<20}{'degenerado':>12}{'reglas':>10}{'TCN':>10}{'rango TCN':>16}")
    for campo, brazos in resultados.items():
        fila = f"{campo:<20}"
        for nombre in ("degenerado", "reglas", "tcn"):
            valores = [r.macro_f1 for r in brazos[nombre]]
            ancho = 12 if nombre == "degenerado" else 10
            fila += f"{np.median(valores):>{ancho}.3f}"
        v = [r.macro_f1 for r in brazos["tcn"]]
        fila += f"{f'{min(v):.3f}-{max(v):.3f}':>16}"
        print(fila)

    print("\nDetalle del campo donde las reglas no pueden llegar (manipulacion):")
    print(resultados["manipulation"]["tcn"][0].format("  pliegue 1, TCN:"))

    artefacto = {
        "fecha": date.today().isoformat(),
        "hz": HZ,
        "epocas": args.epocas,
        "parametros": parametros,
        "pliegues": [p.name for p in pliegues],
        "f1_macro": {
            campo: {
                brazo: [r.macro_f1 for r in informes] for brazo, informes in brazos.items()
            }
            for campo, brazos in resultados.items()
        },
    }
    destino = RAIZ / f"artifacts/tcn-{date.today().isoformat().replace('-', '')}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(artefacto, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nartefacto: {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
