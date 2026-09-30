"""UW-IOM: empareja cada esqueleto con su etiqueta y con su tiempo real.

Todo lo que este módulo sabe del dataset está medido, no supuesto, y la
procedencia de cada afirmación está en `data/README.md`:

  · Las etiquetas están alineadas al FINAL de la secuencia. Hay entre 28 y 109
    etiquetas menos que fotogramas, y el hueco está al principio. Se determinó
    comparando cuánto más inclinado está el tronco en los fotogramas `bend` que en
    los `stand` bajo cada hipótesis: 21,75 grados al final contra 4,11 al
    principio, ganando en 18 de 20 sujetos.
  · El número de fotogramas del vídeo coincide EXACTAMENTE con el de marcas de
    tiempo en los 20 sujetos. Esa correspondencia uno a uno es lo que permite
    fechar cada esqueleto, y se comprueba al cargar en vez de confiarse.
  · El fps de la cabecera del vídeo NO sirve: dice 8, 10, 11 o 12 redondos
    mientras la captura real va de 7,81 a 10,58, y la diferencia no es
    sistemática. La tasa real sale de las marcas de tiempo.
  · El esqueleto 3D del Kinect del sujeto 3 está defectuoso. Su vídeo está bien,
    así que el sujeto se usa por la vía del vídeo y solo queda fuera de lo que se
    calcule sobre Kinect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..pose.schema import SkeletonSequence

RAIZ = Path(__file__).resolve().parents[2]
BASE = RAIZ / "data/raw/uw-iom/xwzzkxtf9s-2/UW IOM Dataset"
KEYPOINTS = RAIZ / "data/keypoints/uwiom"

SUBJECTS = tuple(range(1, 21))

# Sujetos cuyo esqueleto 3D del Kinect no es utilizable. Diagnosticado con
# scripts/probe_subject3.py: su vídeo sí sirve.
KINECT_BROKEN = frozenset({3})

# Los cuatro campos de cada etiqueta, en el orden en que vienen en el fichero.
LABEL_FIELDS = ("object", "motion", "manipulation", "height")

# Orden de articulaciones del SDK del Kinect v2, comprobado contra los datos
# midiendo la inclinación del tronco (p50 de 8,1° de pie, p95 de 69,1° doblado).
KINECT_SPINE_BASE, KINECT_SPINE_SHOULDER = 0, 20


@dataclass(frozen=True)
class Subject:
    """Un participante, ya recortado a la parte que tiene etiqueta.

    skeleton:   la secuencia de esqueletos, solo los fotogramas etiquetados.
    labels:     una tupla de cuatro campos por fotograma.
    timestamps: el tiempo de cada fotograma, en las unidades del dataset.
    """

    index: int
    skeleton: SkeletonSequence
    labels: list[tuple[str, str, str, str]]
    timestamps: np.ndarray

    def __post_init__(self) -> None:
        if not (len(self.skeleton) == len(self.labels) == len(self.timestamps)):
            raise ValueError(
                f"sujeto {self.index}: esqueleto {len(self.skeleton)}, etiquetas "
                f"{len(self.labels)}, tiempos {len(self.timestamps)} no cuadran"
            )

    @property
    def fps(self) -> float:
        """La tasa REAL de captura, de las marcas de tiempo. No la del contenedor."""
        span = float(self.timestamps[-1] - self.timestamps[0])
        return (len(self.timestamps) - 1) / span if span > 0 else float("nan")

    def field(self, name: str) -> np.ndarray:
        """Un campo de la etiqueta como array, p. ej. `field("motion")`."""
        position = LABEL_FIELDS.index(name)
        return np.array([label[position] for label in self.labels])


def parse_label(line: str) -> tuple[str, str, str, str]:
    """`box_bend_pick-up_low` -> ("box", "bend", "pick-up", "low")."""
    parts = line.strip().split("_")
    if len(parts) != len(LABEL_FIELDS):
        raise ValueError(f"etiqueta con {len(parts)} campos en vez de 4: {line!r}")
    return parts[0], parts[1], parts[2], parts[3]


def read_labels(index: int) -> list[tuple[str, str, str, str]]:
    path = BASE / "VideoLabels" / f"{index:02d}.txt"
    return [
        parse_label(line)
        for line in path.read_text(encoding="utf-8", errors="replace").split("\n")
        if line.strip()
    ]


def read_video_timestamps(index: int) -> np.ndarray:
    """Las marcas de tiempo de los fotogramas de vídeo, del fichero del Kinect.

    Se leen del `.mat` aunque el esqueleto venga del vídeo, porque es la única
    fuente de la tasa real. Nótese la trampa de nombres del dataset: los `.mat`
    van sin cero a la izquierda (`3.mat`) y los `.txt` con él (`03.txt`).
    """
    import h5py

    with h5py.File(BASE / "JointPositions" / f"{index}.mat", "r") as f:
        return np.array(f["videotimelogger"]).ravel()


def read_kinect_skeleton(index: int) -> np.ndarray:
    """El esqueleto 3D del Kinect, (T, 25, 3) en metros. Referencia, no producción."""
    import h5py

    if index in KINECT_BROKEN:
        raise ValueError(
            f"el esqueleto de Kinect del sujeto {index} está defectuoso "
            "(ver data/README.md). Usa la vía del vídeo."
        )
    with h5py.File(BASE / "JointPositions" / f"{index}.mat", "r") as f:
        return np.array(f["bodylogger3D"])


def load_subject(index: int) -> Subject:
    """Carga un sujeto con sus esqueletos, etiquetas y tiempos ya alineados.

    Comprueba al cargar las dos cosas de las que depende la alineación —que haya
    una marca de tiempo por fotograma, y que las etiquetas no sean más que los
    fotogramas— porque si alguna dejara de cumplirse el dataset saldría desplazado
    sin dar ningún error.
    """
    skeleton = SkeletonSequence.load(KEYPOINTS / f"{index:02d}.npz")
    labels = read_labels(index)
    timestamps = read_video_timestamps(index)

    if len(skeleton) != len(timestamps):
        raise ValueError(
            f"sujeto {index}: {len(skeleton)} fotogramas extraídos y "
            f"{len(timestamps)} marcas de tiempo. La correspondencia uno a uno es "
            "lo que sostiene la alineación; sin ella no se puede fechar nada."
        )
    offset = len(skeleton) - len(labels)
    if offset < 0:
        raise ValueError(
            f"sujeto {index}: {len(labels)} etiquetas para {len(skeleton)} "
            "fotogramas. Se esperaba que las etiquetas fueran menos."
        )

    corte = slice(offset, offset + len(labels))
    return Subject(
        index=index,
        skeleton=SkeletonSequence(
            keypoints=skeleton.keypoints[corte],
            scores=skeleton.scores[corte],
            boxes=skeleton.boxes[corte],
            present=skeleton.present[corte],
            meta=skeleton.meta | {"label_offset": offset, "subject": index},
        ),
        labels=labels,
        timestamps=timestamps[corte],
    )


def load_all(subjects=SUBJECTS) -> list[Subject]:
    return [load_subject(i) for i in subjects]
