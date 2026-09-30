"""El arnés: todos los brazos se miden sobre EXACTAMENTE los mismos fotogramas.

Existe por un detalle que arruinaría la comparación sin dar ningún error. Las
reglas predicen cada fotograma por separado; un modelo temporal necesita dos
segundos de pasado, así que **no puede predecir los primeros veinte fotogramas de
cada sujeto**. Si cada brazo se evaluara sobre lo que puede, estarían midiéndose
sobre conjuntos distintos, y la diferencia entre sus números incluiría esa
diferencia de material.

No es un matiz pequeño: los fotogramas del principio son los de la persona
entrando en escena y colocándose, que son más fáciles —casi todo `walk` y `stand`
sin manipular—. Quitárselos al modelo y dejárselos a las reglas le regalaría a las
reglas un puñado de aciertos baratos.

Así que aquí se calcula UNA lista de fotogramas evaluables por sujeto, y todos los
brazos predicen sobre ella. Un brazo que no pueda cubrirla entera se rechaza.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..baseline.reba import body_angles
from ..datasets.uwiom import LABEL_FIELDS, load_subject
from ..datasets.windows import resample
from .metrics import Report, report

HZ = 10.0
WINDOW_SECONDS = 2.0


@dataclass(frozen=True)
class SubjectData:
    """Un sujeto ya remuestreado, sin fotogramas vacíos, con su ventana de contexto.

    keypoints:  (N, 17, 2) todos los fotogramas con persona
    labels:     (N, 4)
    evaluable:  índices desde los que hay ventana completa hacia atrás. Es el
                conjunto sobre el que se miden TODOS los brazos.
    """

    index: int
    keypoints: np.ndarray
    trunk_flexion: np.ndarray
    labels: np.ndarray
    evaluable: np.ndarray
    window_frames: int

    def windows(self) -> np.ndarray:
        """(M, T, 17, 2) — una ventana causal por fotograma evaluable."""
        t = self.window_frames
        return np.stack([self.keypoints[i - t + 1 : i + 1] for i in self.evaluable])

    def y(self, field: str | None = None) -> np.ndarray:
        etiquetas = self.labels[self.evaluable]
        if field is None:
            return np.array(["_".join(f) for f in etiquetas])
        return etiquetas[:, LABEL_FIELDS.index(field)]


def load(index: int, hz: float = HZ, window_seconds: float = WINDOW_SECONDS) -> SubjectData:
    s = load_subject(index)
    elegidos, _, presente = resample(s, hz)
    keypoints = s.skeleton.keypoints[elegidos][presente]
    labels = np.array(s.labels)[elegidos][presente]

    t = int(round(window_seconds * hz))
    if len(keypoints) <= t:
        raise ValueError(f"sujeto {index}: no caben ventanas de {window_seconds}s")
    return SubjectData(
        index=index,
        keypoints=keypoints,
        trunk_flexion=body_angles(keypoints)["trunk_flexion"],
        labels=labels,
        evaluable=np.arange(t - 1, len(keypoints)),
        window_frames=t,
    )


def evaluate(subjects: list[SubjectData], predictions: list[dict[str, np.ndarray]]) -> dict[str, Report]:
    """Puntúa un brazo sobre los fotogramas evaluables, campo a campo.

    `predictions[i][campo]` tiene que tener exactamente tantas entradas como
    fotogramas evaluables tenga `subjects[i]`. Se comprueba: un brazo que devuelva
    de más o de menos está prediciendo sobre otro material, y el número que saliera
    no sería comparable con nada.
    """
    salida = {}
    for campo in (*LABEL_FIELDS, None):
        nombre = campo or "etiqueta completa"
        y_true, y_pred = [], []
        for sujeto, prediccion in zip(subjects, predictions):
            esperados = len(sujeto.evaluable)
            if campo is None:
                columnas = np.stack([prediccion[c] for c in LABEL_FIELDS], axis=1)
                valores = np.array(["_".join(f) for f in columnas])
            else:
                valores = np.asarray(prediccion[campo])
            if len(valores) != esperados:
                raise ValueError(
                    f"sujeto {sujeto.index}, campo {nombre}: el brazo devolvió "
                    f"{len(valores)} predicciones para {esperados} fotogramas "
                    "evaluables. Los brazos tienen que cubrir el mismo material."
                )
            y_true.append(sujeto.y(campo))
            y_pred.append(valores)
        salida[nombre] = report(np.concatenate(y_true), np.concatenate(y_pred))
    return salida
