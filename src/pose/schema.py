"""El formato de una secuencia de esqueletos, y las normalizaciones que admite.

Este módulo es el contrato entre la extracción y todo lo que viene después.

DOS DECISIONES QUE GOBIERNAN EL RESTO

1. Lo que se guarda son PÍXELES CRUDOS, no coordenadas normalizadas.
   La normalización es una decisión del entrenamiento y cambia varias veces a lo
   largo de un proyecto; la extracción cuesta horas de GPU y no se puede repetir
   a la ligera. Cocinar los datos al extraerlos obliga a re-extraer cada vez que
   se cambia de idea. Aquí el `.npz` es la verdad cruda y la normalización se
   aplica al cargar.

2. La normalización para el modelo tiene que ser ISOTRÓPICA, y esto no es un
   detalle: el riesgo ergonómico se mide en ÁNGULOS del cuerpo. Dividir la x por
   el ancho de la caja y la y por el alto —que es lo que se hace por costumbre—
   deforma los ángulos, porque estira el cuerpo en un eje más que en el otro. Un
   tronco a 45 grados dentro de una caja alta y estrecha deja de medir 45 grados.
   Por eso las dos coordenadas se dividen por la MISMA escala.

PRIVACIDAD: aquí no hay ningún camino por el que una imagen llegue al disco. Una
secuencia son articulaciones y nada más. Es estructural, no una promesa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Articulaciones en el orden de COCO, que es el que devuelve YOLO-pose.
JOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
N_JOINTS = len(JOINT_NAMES)

JOINT = {name: i for i, name in enumerate(JOINT_NAMES)}

# Versión del formato. Si cambia la semántica de lo guardado, sube, y el cargador
# se niega a leer lo viejo en silencio: un dataset mezclado no se ve venir.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SkeletonSequence:
    """Una persona a lo largo de T fotogramas.

    keypoints: (T, 17, 2) en píxeles de la imagen original.
    scores:    (T, 17) confianza por articulación, tal como la da el modelo.
    boxes:     (T, 4) caja de la persona en píxeles, formato xyxy.
    present:   (T,) bool. False donde el modelo no encontró a la persona; en esos
               fotogramas keypoints y scores son cero y NO deben interpolarse sin
               decirlo, porque un hueco tapado es una postura inventada.
    meta:      procedencia. De dónde salió, con qué modelo y cuándo.
    """

    keypoints: np.ndarray
    scores: np.ndarray
    boxes: np.ndarray
    present: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = self.keypoints.shape[0]
        if self.keypoints.shape != (t, N_JOINTS, 2):
            raise ValueError(
                f"keypoints debe ser (T, {N_JOINTS}, 2), llegó {self.keypoints.shape}"
            )
        if self.scores.shape != (t, N_JOINTS):
            raise ValueError(f"scores debe ser (T, {N_JOINTS}), llegó {self.scores.shape}")
        if self.boxes.shape != (t, 4):
            raise ValueError(f"boxes debe ser (T, 4), llegó {self.boxes.shape}")
        if self.present.shape != (t,):
            raise ValueError(f"present debe ser (T,), llegó {self.present.shape}")

    def __len__(self) -> int:
        return self.keypoints.shape[0]

    # --- persistencia ---------------------------------------------------------

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            keypoints=self.keypoints.astype(np.float32),
            scores=self.scores.astype(np.float32),
            boxes=self.boxes.astype(np.float32),
            present=self.present.astype(bool),
            meta=np.array(repr(self.meta | {"schema_version": SCHEMA_VERSION})),
        )

    @classmethod
    def load(cls, path) -> "SkeletonSequence":
        import ast

        with np.load(path, allow_pickle=False) as z:
            meta = ast.literal_eval(str(z["meta"]))
            version = meta.get("schema_version")
            if version != SCHEMA_VERSION:
                raise ValueError(
                    f"{path} usa el formato {version} y este código lee el "
                    f"{SCHEMA_VERSION}. Re-extraer, no mezclar."
                )
            return cls(
                keypoints=z["keypoints"],
                scores=z["scores"],
                boxes=z["boxes"],
                present=z["present"],
                meta=meta,
            )


# --- normalización -----------------------------------------------------------


def normalize_isotropic(
    keypoints: np.ndarray, boxes: np.ndarray
) -> np.ndarray:
    """Centra en la cadera y escala por la ALTURA de la caja, en los dos ejes.

    Devuelve (T, 17, 2). Invariante a dónde está la persona en la imagen, a la
    distancia a la cámara y a la resolución del vídeo — y, porque la escala es la
    misma en x y en y, PRESERVA LOS ÁNGULOS del cuerpo.

    La altura de la caja no es una escala perfecta: al doblarse, la proyección en
    2D de una persona cambia y con ella la caja. Ninguna escala medida sobre la
    imagen se libra de eso. Lo que importa aquí es que sea isotrópica; cuál se
    elige es una decisión que se puede cambiar sin volver a extraer nada, que es
    justo el motivo de guardar píxeles crudos.
    """
    hips = keypoints[:, [JOINT["left_hip"], JOINT["right_hip"]], :].mean(axis=1)
    height = (boxes[:, 3] - boxes[:, 1]).astype(np.float64)
    # Una caja de altura cero pasaría a dividir por cero y llenaría la secuencia
    # de infinitos que reventarían el entrenamiento mil pasos más tarde.
    height = np.where(height > 1.0, height, 1.0)
    return ((keypoints - hips[:, None, :]) / height[:, None, None]).astype(np.float32)


def joint_angle(
    keypoints: np.ndarray, a: str, b: str, c: str
) -> np.ndarray:
    """Ángulo en grados del vértice `b`, entre los segmentos b→a y b→c.

    Es la operación de la que cuelga el cálculo de riesgo postural, así que vive
    aquí y no dentro del módulo de REBA: la normalización tiene que poder
    comprobarse contra ella.
    """
    ba = keypoints[:, JOINT[a], :] - keypoints[:, JOINT[b], :]
    bc = keypoints[:, JOINT[c], :] - keypoints[:, JOINT[b], :]
    norms = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    cos = np.divide(
        (ba * bc).sum(axis=1), norms, out=np.zeros(len(keypoints)), where=norms > 1e-9
    )
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
