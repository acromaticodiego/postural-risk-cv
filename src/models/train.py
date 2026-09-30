"""Entrenar el TCN en un pliegue. Vive aquí para que lo usen varios guiones.

Estaba dentro del guion de entrenamiento y lo necesitaba también el del informe por
tareas. Duplicarlo habría sido peor que moverlo: dos copias del bucle de
entrenamiento divergen, y entonces dos números que se presentan juntos salen de
modelos distintos sin que nadie lo note.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..datasets.uwiom import LABEL_FIELDS
from .tcn import SkeletonTCN, class_weights, normalize_windows

SEED = 20260929


def train_fold(
    train_subjects: list,
    test_subjects: list,
    epochs: int = 30,
    device: str = "cuda",
    verbose: bool = False,
) -> tuple[list[dict[str, np.ndarray]], int]:
    """Entrena con los sujetos de entrenamiento y predice sobre los de prueba.

    La semilla se fija para que dos corridas del mismo pliegue den lo mismo: sin
    eso, comparar dos versiones del modelo mezcla la diferencia real con el azar de
    la inicialización.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x_train = normalize_windows(np.concatenate([s.windows() for s in train_subjects]))
    y_train = {c: np.concatenate([s.y(c) for s in train_subjects]) for c in LABEL_FIELDS}
    classes = {c: sorted(set(y_train[c].tolist())) for c in LABEL_FIELDS}
    index = {c: {v: i for i, v in enumerate(classes[c])} for c in LABEL_FIELDS}

    xt = torch.tensor(x_train, device=device)
    yt = {
        c: torch.tensor([index[c][v] for v in y_train[c]], device=device) for c in LABEL_FIELDS
    }

    model = SkeletonTCN({c: len(classes[c]) for c in LABEL_FIELDS}).to(device)
    losses = {
        c: nn.CrossEntropyLoss(weight=class_weights(y_train[c], classes[c]).to(device))
        for c in LABEL_FIELDS
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n, batch = len(xt), 256
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch):
            chunk = order[start : start + batch]
            out = model(xt[chunk])
            loss = sum(losses[c](out[c], yt[c][chunk]) for c in LABEL_FIELDS)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(chunk)
        scheduler.step()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"      epoca {epoch + 1:>3}  perdida {total / n:.4f}")

    model.eval()
    predictions = []
    with torch.no_grad():
        for subject in test_subjects:
            x = torch.tensor(normalize_windows(subject.windows()), device=device)
            out = model(x)
            predictions.append(
                {
                    c: np.array([classes[c][i] for i in out[c].argmax(dim=1).cpu().numpy()])
                    for c in LABEL_FIELDS
                }
            )
    return predictions, model.n_parameters
