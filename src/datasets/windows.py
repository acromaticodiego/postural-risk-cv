"""De una secuencia entera a las ventanas con las que se entrena.

DOS DECISIONES QUE DECIDEN SI EL RESULTADO ES DESPLEGABLE

1. LAS VENTANAS SON CAUSALES: una ventana termina en el fotograma que etiqueta y
   solo contiene pasado. Lo cómodo es centrarla —el fotograma del medio suele ser
   más fácil de clasificar porque se ve cómo empieza y cómo acaba el movimiento—
   pero eso usa fotogramas del FUTURO, y un sistema que necesita el futuro no
   funciona en vivo. Entrenar con ventanas centradas y luego prometer tiempo real
   es una de las formas más comunes de publicar un número que no se puede
   reproducir en producción. Se puede pedir el modo centrado explícitamente, y
   entonces queda anotado que el resultado no es desplegable en vivo.

2. EL TIEMPO SE MIDE EN SEGUNDOS, NO EN FOTOGRAMAS. En UW-IOM la tasa real de
   captura va de 7,81 a 10,58 según el sujeto, así que una ventana de 30
   fotogramas duraría 3,8 segundos en un participante y 2,8 en otro: el sujeto se
   convertiría en una variable oculta del experimento. Se remuestrea a una tasa
   común usando las marcas de tiempo del dataset, que existen fotograma a
   fotograma.

Y una tercera, menor pero que también sesga: la etiqueta de una ventana es la del
fotograma que la cierra, no la mayoritaria. La mayoritaria suaviza las
transiciones y hace que el número suba, porque borra justo los momentos en que la
acción cambia, que son los difíciles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pose.schema import N_JOINTS
from .uwiom import LABEL_FIELDS, Subject

DEFAULT_HZ = 10.0
DEFAULT_WINDOW_SECONDS = 2.0
DEFAULT_STRIDE_SECONDS = 0.5


@dataclass(frozen=True)
class WindowSet:
    """Ventanas listas para un modelo, con de dónde salió cada una.

    keypoints: (N, T, 17, 2)
    labels:    (N, 4) los cuatro campos de la etiqueta del fotograma final
    subjects:  (N,) de qué participante viene cada ventana, para que la partición
               por sujeto se pueda comprobar aguas abajo y no solo prometer
    complete:  (N,) False si a la ventana le faltaba algún fotograma con persona
    """

    keypoints: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    complete: np.ndarray
    hz: float
    window_seconds: float
    causal: bool

    def __len__(self) -> int:
        return len(self.keypoints)

    def field(self, name: str) -> np.ndarray:
        return self.labels[:, LABEL_FIELDS.index(name)]

    def filter_subjects(self, subjects) -> "WindowSet":
        """El subconjunto de unos participantes. Es lo que consume la partición."""
        mask = np.isin(self.subjects, list(subjects))
        return WindowSet(
            keypoints=self.keypoints[mask],
            labels=self.labels[mask],
            subjects=self.subjects[mask],
            complete=self.complete[mask],
            hz=self.hz,
            window_seconds=self.window_seconds,
            causal=self.causal,
        )


def resample(subject: Subject, hz: float = DEFAULT_HZ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lleva un sujeto a una tasa fija usando sus marcas de tiempo reales.

    Los keypoints se toman por VECINO MÁS CERCANO, no interpolados. Interpolar
    entre dos posturas inventa una postura intermedia que nadie adoptó, y aquí las
    posturas son el dato; además, con tasas de 8 a 10 Hz llevadas a 10 Hz, el
    vecino más cercano está a menos de 60 ms.

    Devuelve (índices elegidos, tiempos nuevos, presencia).
    """
    tiempos = subject.timestamps.astype(np.float64)
    inicio, fin = tiempos[0], tiempos[-1]
    if not np.isfinite(inicio) or fin <= inicio:
        raise ValueError(f"sujeto {subject.index}: marcas de tiempo inservibles")

    nuevos = np.arange(inicio, fin, 1.0 / hz)
    elegidos = np.abs(tiempos[None, :] - nuevos[:, None]).argmin(axis=1)
    return elegidos, nuevos, subject.skeleton.present[elegidos]


def windows_from_subject(
    subject: Subject,
    hz: float = DEFAULT_HZ,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
    causal: bool = True,
) -> WindowSet:
    elegidos, _, presente = resample(subject, hz)
    keypoints = subject.skeleton.keypoints[elegidos]
    etiquetas = np.array(subject.labels)[elegidos]

    largo = int(round(window_seconds * hz))
    paso = max(1, int(round(stride_seconds * hz)))
    if largo < 2:
        raise ValueError(f"ventana de {window_seconds}s a {hz}Hz son {largo} fotogramas")
    if largo > len(keypoints):
        raise ValueError(
            f"sujeto {subject.index}: la ventana de {window_seconds}s no cabe en la secuencia"
        )

    trozos, etq, completas = [], [], []
    for fin in range(largo, len(keypoints) + 1, paso):
        corte = slice(fin - largo, fin)
        trozos.append(keypoints[corte])
        # Causal: la etiqueta es la del último fotograma, que es el presente que
        # el sistema tendría que decidir en vivo. Centrada: la del medio, que usa
        # futuro y por tanto no es desplegable en tiempo real.
        indice = fin - 1 if causal else fin - largo // 2 - 1
        etq.append(etiquetas[indice])
        completas.append(bool(presente[corte].all()))

    return WindowSet(
        keypoints=np.asarray(trozos, dtype=np.float32).reshape(-1, largo, N_JOINTS, 2),
        labels=np.asarray(etq),
        subjects=np.full(len(trozos), subject.index),
        complete=np.asarray(completas, dtype=bool),
        hz=hz,
        window_seconds=window_seconds,
        causal=causal,
    )


def windows_from_subjects(subjects: list[Subject], **kwargs) -> WindowSet:
    partes = [windows_from_subject(s, **kwargs) for s in subjects]
    primera = partes[0]
    return WindowSet(
        keypoints=np.concatenate([p.keypoints for p in partes]),
        labels=np.concatenate([p.labels for p in partes]),
        subjects=np.concatenate([p.subjects for p in partes]),
        complete=np.concatenate([p.complete for p in partes]),
        hz=primera.hz,
        window_seconds=primera.window_seconds,
        causal=primera.causal,
    )
