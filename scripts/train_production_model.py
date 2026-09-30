"""Entrena el modelo que corre en vivo y lo guarda.

Los pliegues de validación cruzada sirven para MEDIR: cada uno entrena con doce
sujetos y se mide con otros cuatro. El modelo que se despliega es otro: se entrena
con **los dieciséis sujetos de desarrollo**, porque no hay razón para tirar datos
cuando ya no se va a medir con ellos.

Y por eso este modelo NO tiene un número propio que publicar. El número publicable
es el de la validación cruzada —F1 macro 0,870 en movimiento, 0,741 en
manipulación— que se obtuvo con modelos entrenados con menos datos, y por tanto es
si acaso pesimista respecto a este. Presentar una cifra medida sobre los mismos
sujetos con los que se entrenó sería el error más viejo del oficio.

El reservado (11, 14, 19, 20) sigue sin tocarse.

    .\\.venv\\Scripts\\python.exe scripts\\train_production_model.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.uwiom import LABEL_FIELDS  # noqa: E402
from src.eval.harness import HZ, WINDOW_SECONDS, load  # noqa: E402
from src.eval.splits import development_subjects, holdout_subjects  # noqa: E402
from src.models.tcn import SkeletonTCN, class_weights, normalize_windows  # noqa: E402
from src.models.train import SEED  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "artifacts/modelo"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sujetos = development_subjects()
    print(f"Entrenando con {len(sujetos)} sujetos: {list(sujetos)}")
    print(f"Reservado intacto: {list(holdout_subjects())}\n")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    datos = [load(i) for i in sujetos]
    x = normalize_windows(np.concatenate([s.windows() for s in datos]))
    y = {c: np.concatenate([s.y(c) for s in datos]) for c in LABEL_FIELDS}
    clases = {c: sorted(set(y[c].tolist())) for c in LABEL_FIELDS}
    indices = {c: {v: i for i, v in enumerate(clases[c])} for c in LABEL_FIELDS}

    xt = torch.tensor(x, device=device)
    yt = {c: torch.tensor([indices[c][v] for v in y[c]], device=device) for c in LABEL_FIELDS}

    modelo = SkeletonTCN({c: len(clases[c]) for c in LABEL_FIELDS}).to(device)
    perdidas = {
        c: nn.CrossEntropyLoss(weight=class_weights(y[c], clases[c]).to(device))
        for c in LABEL_FIELDS
    }
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=2e-3, weight_decay=1e-4)
    epocas = 30
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
        if (epoca + 1) % 10 == 0:
            print(f"  epoca {epoca + 1:>3}  perdida {total / n:.4f}")

    DESTINO.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), DESTINO / "tcn.pt")
    (DESTINO / "tcn.json").write_text(
        json.dumps(
            {
                "entrenado": date.today().isoformat(),
                "sujetos": list(sujetos),
                "reservado_intacto": list(holdout_subjects()),
                "hz": HZ,
                "window_seconds": WINDOW_SECONDS,
                "epocas": epocas,
                "clases": clases,
                "ventanas": int(n),
                "parametros": modelo.n_parameters,
                "nota": (
                    "Entrenado con TODOS los sujetos de desarrollo. No tiene metrica "
                    "propia: la publicable es la de la validacion cruzada."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n{n:,} ventanas · {modelo.n_parameters:,} parametros")
    print(f"guardado en {DESTINO.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
