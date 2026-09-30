"""Un TCN causal sobre secuencias de esqueletos, con cuatro cabezas.

POR QUÉ ESTA ARQUITECTURA Y NO UNA RED RECURRENTE: un TCN ve toda la ventana de
golpe en vez de paso a paso, así que entrena en segundos en una 3050 y su campo
receptivo es una cuenta, no una esperanza. Para dos segundos de contexto a 10 Hz
—veinte fotogramas— basta con tres bloques dilatados.

CAUSAL, y no por gusto: cada convolución solo mira hacia atrás. Es lo que hace que
el modelo se pueda desplegar en vivo, y tiene que ser coherente con las ventanas,
que también son causales. Un modelo que mire el futuro daría mejores números y no
se podría instalar en ninguna planta.

CUATRO CABEZAS sobre un tronco común: el dataset etiqueta objeto, movimiento,
manipulación y altura, y son cuatro preguntas sobre el mismo movimiento. Compartir
la representación es más barato y además evita el problema de tratar las 17
combinaciones como clases independientes, donde las raras no tienen apenas
ejemplos.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..pose.schema import JOINT, N_JOINTS


def normalize_windows(windows: np.ndarray) -> np.ndarray:
    """(M, T, 17, 2) -> (M, T, 34), centrado en la cadera y a escala isotrópica.

    La escala es la altura del cuerpo en ese fotograma y se aplica IGUAL a las dos
    coordenadas. Dividir cada eje por su lado de la caja deformaría los ángulos, y
    los ángulos son lo que distingue una postura peligrosa de una segura.
    """
    caderas = windows[:, :, [JOINT["left_hip"], JOINT["right_hip"]], :].mean(axis=2)
    alto = windows[:, :, :, 1].max(axis=2) - windows[:, :, :, 1].min(axis=2)
    alto = np.maximum(alto, 1.0)[:, :, None, None]
    centrado = (windows - caderas[:, :, None, :]) / alto
    return centrado.reshape(len(windows), windows.shape[1], N_JOINTS * 2).astype(np.float32)


class CausalBlock(nn.Module):
    """Convolución dilatada que solo mira hacia atrás, con atajo residual."""

    def __init__(self, channels: int, dilation: int, kernel: int = 3, dropout: float = 0.2):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=self.padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=self.padding)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # El padding se pone a los dos lados y se recorta el final: así ningún paso
        # ve fotogramas posteriores al suyo. Es la parte que hace la red causal, y
        # quitarla por descuido no daría error, solo números mejores y falsos.
        y = self.conv1(x)[:, :, : -self.padding]
        y = self.drop(self.act(self.norm1(y)))
        y = self.conv2(y)[:, :, : -self.padding]
        y = self.drop(self.act(self.norm2(y)))
        return self.act(x + y)


class SkeletonTCN(nn.Module):
    def __init__(self, n_classes: dict[str, int], channels: int = 64, dilations=(1, 2, 4)):
        super().__init__()
        self.entrada = nn.Conv1d(N_JOINTS * 2, channels, 1)
        self.bloques = nn.Sequential(*[CausalBlock(channels, d) for d in dilations])
        self.cabezas = nn.ModuleDict(
            {campo: nn.Linear(channels, n) for campo, n in n_classes.items()}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (batch, T, 34) -> (batch, 34, T)
        h = self.bloques(self.entrada(x.transpose(1, 2)))
        # Se lee el ÚLTIMO paso temporal: es el presente que el sistema tendría que
        # decidir en vivo, y es el fotograma que lleva la etiqueta de la ventana.
        ultimo = h[:, :, -1]
        return {campo: cabeza(ultimo) for campo, cabeza in self.cabezas.items()}

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def class_weights(y: np.ndarray, classes: list[str]) -> torch.Tensor:
    """Pesos inversos a la frecuencia, porque la métrica es F1 macro.

    Sin esto, la red aprende a contestar `stand / place` casi siempre —que es el 35%
    del material— y sale con un *accuracy* presentable y un F1 macro de risa. El
    desbalance del dataset es de 30 a 1 entre la clase más común y la más rara.
    """
    cuentas = np.array([max((y == c).sum(), 1) for c in classes], dtype=np.float64)
    pesos = cuentas.sum() / (len(classes) * cuentas)
    return torch.tensor(pesos, dtype=torch.float32)
