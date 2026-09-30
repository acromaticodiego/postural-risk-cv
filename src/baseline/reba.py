"""REBA a partir de un esqueleto. Es la LÍNEA BASE del proyecto, no el producto.

REBA (Rapid Entire Body Assessment, Hignett y McAtamney, 2000) es la norma con la
que un ingeniero de seguridad y salud en el trabajo puntúa una postura hoy, a mano
y con un portapapeles. Automatizarla es el punto de partida: si el modelo que se
entrena después no aporta nada sobre esto, el proyecto no tiene modelo, tiene una
calculadora — y saberlo desde el principio vale más que descubrirlo al final.

QUÉ SE PUEDE Y QUÉ NO SE PUEDE MEDIR CON UNA CÁMARA

REBA completo NO es calculable con una sola cámara y un esqueleto 2D, y decirlo es
parte del trabajo. De los componentes de la norma:

  OBSERVABLES        tronco, cuello, piernas, brazo, antebrazo — ángulos entre
                     articulaciones que el esqueleto sí da.
  NO OBSERVABLES     la muñeca (el esqueleto COCO acaba en la muñeca: no hay
                     mano ni dedos), la torsión y la inclinación lateral del
                     tronco y del cuello (necesitan profundidad para
                     distinguirse de la proyección), el peso de la carga, y la
                     calidad del agarre.
  DERIVABLE DEL      el ajuste de actividad: repeticiones por minuto y posturas
  TIEMPO             sostenidas. Requiere la secuencia, no el fotograma, y es el
                     único componente que un sistema que puntúe imágenes sueltas
                     no puede calcular.

Los no observables son ASUNCIONES declaradas en `RebaAssumptions`, con el valor
neutro de la norma por defecto, y quién las fija es una decisión de producto: cada
una mueve el puntaje final y por tanto el nivel de riesgo que se reporta.

Y UNA ADVERTENCIA MEDIDA: los ángulos de un esqueleto 2D son ÁNGULOS PROYECTADOS,
y exageran. Medido el 29/09 sobre UW-IOM, la separación de inclinación del tronco
entre agacharse y estar de pie sale en +30,1 grados con la cámara y en +14,7 con
el esqueleto 3D del sensor de profundidad en el mismo sujeto. Como REBA corta por
grados, exagerar sube el nivel declarado. Cuantificar eso es una de las preguntas
del proyecto, no un detalle a tapar.

PROCEDENCIA DE LAS TABLAS: transcritas de una implementación pública que declara
seguir Hignett y McAtamney (2000), y coinciden con los valores de uso común. NO se
han contrastado contra la publicación original —el acceso falló—, así que antes de
publicar cualquier cifra hay que verificarlas contra la hoja oficial. Hay una
celda concretamente dudosa, marcada abajo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pose.schema import JOINT

# --- las tablas de la norma --------------------------------------------------

# Tabla A: fila = tronco (1-5); columna = cuello (1-3) x piernas (1-4) aplanados,
# o sea columna = (cuello - 1) * 4 + (piernas - 1).
TABLE_A = np.array(
    [
        [1, 2, 3, 4, 1, 2, 3, 4, 3, 3, 5, 6],
        [2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7],
        [2, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8],
        [3, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9],
        # OJO: la última fila aparece como 7,8,9,9 en la fuente usada y como
        # 7,7,8,9 en otras transcripciones de uso común. Afecta a cuello 3 con
        # tronco 5, que es la postura más extrema de la tabla. Pendiente de
        # verificar contra la publicación original.
        [4, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 9],
    ]
)

# Tabla B: fila = brazo (1-6); columna = antebrazo (1-2) x muñeca (1-3),
# o sea columna = (antebrazo - 1) * 3 + (muñeca - 1).
TABLE_B = np.array(
    [
        [1, 2, 2, 1, 2, 3],
        [1, 2, 3, 2, 3, 4],
        [3, 4, 5, 4, 5, 5],
        [4, 5, 5, 5, 6, 7],
        [6, 7, 8, 7, 8, 8],
        [7, 8, 8, 8, 9, 9],
    ]
)

# Tabla C: fila = puntaje A (1-12), columna = puntaje B (1-12).
TABLE_C = np.array(
    [
        [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
        [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
        [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
        [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
        [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
        [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
        [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
        [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
        [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
        [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
        [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
        [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    ]
)

# Niveles de acción de la norma. El corte a partir del cual el sistema CUENTA una
# exposición es una decisión de producto aparte: la norma describe niveles, no
# decide cuándo suena una alarma.
ACTION_LEVELS = (
    (1, 1, "despreciable"),
    (2, 3, "bajo"),
    (4, 7, "medio"),
    (8, 10, "alto"),
    (11, 15, "muy alto"),
)


def action_level(score: int) -> str:
    for low, high, name in ACTION_LEVELS:
        if low <= score <= high:
            return name
    raise ValueError(f"puntaje REBA fuera de rango: {score}")


# --- los componentes no observables, declarados como lo que son --------------


@dataclass(frozen=True)
class RebaAssumptions:
    """Lo que una cámara monocular no puede ver y hay que suponer.

    Los valores por defecto son los NEUTROS de la norma: los que no penalizan. Eso
    hace que el puntaje resultante sea una COTA INFERIOR del riesgo real, y decirlo
    así es más honesto que elegir valores intermedios que parecen prudentes y no
    son ni una cosa ni la otra.

    Cada campo mueve el puntaje final, así que fijarlos es una decisión de
    producto y no de implementación.
    """

    wrist: int = 1
    """Muñeca (1-3). El esqueleto acaba en la muñeca: no hay mano que medir."""

    trunk_twist: bool = False
    """Torsión o inclinación lateral del tronco. Suma 1 al tronco."""

    neck_twist: bool = False
    """Torsión o inclinación lateral del cuello. Suma 1 al cuello."""

    load_force: int = 0
    """Carga (0: <5 kg, 1: 5-10 kg, 2: >10 kg, +1 si es brusca). Sin kilos, 0."""

    coupling: int = 0
    """Agarre (0 bueno, 1 regular, 2 malo, 3 inaceptable). Sin ver la mano, 0."""

    def __post_init__(self) -> None:
        if not 1 <= self.wrist <= 3:
            raise ValueError(f"muñeca fuera de 1-3: {self.wrist}")
        if not 0 <= self.load_force <= 3:
            raise ValueError(f"carga fuera de 0-3: {self.load_force}")
        if not 0 <= self.coupling <= 3:
            raise ValueError(f"agarre fuera de 0-3: {self.coupling}")


# --- de ángulo a puntaje, según la norma -------------------------------------


def trunk_score(flexion_deg: np.ndarray) -> np.ndarray:
    """Erguido 1; hasta 20° 2; 20-60° 3; más de 60° 4."""
    return np.select(
        [flexion_deg < 5, flexion_deg <= 20, flexion_deg <= 60],
        [1, 2, 3],
        default=4,
    ).astype(int)


def neck_score(flexion_deg: np.ndarray) -> np.ndarray:
    """Hasta 20° 1; más de 20° 2. La extensión también puntúa 2 y no se distingue
    aquí: en 2D, cabeza atrás y cabeza adelante se confunden según el ángulo de
    cámara, así que se cuenta por magnitud."""
    return np.where(np.abs(flexion_deg) <= 20, 1, 2).astype(int)


def legs_score(knee_flexion_deg: np.ndarray, bilateral: np.ndarray) -> np.ndarray:
    """Apoyo bilateral 1, unilateral 2; más 1 si la rodilla flexiona 30-60° y 2 si más."""
    base = np.where(bilateral, 1, 2)
    extra = np.select([knee_flexion_deg < 30, knee_flexion_deg <= 60], [0, 1], default=2)
    return (base + extra).astype(int)


def upper_arm_score(elevation_deg: np.ndarray) -> np.ndarray:
    """Hasta 20° 1; 20-45° 2; 45-90° 3; más de 90° 4.

    Sin los ajustes de la norma por hombro elevado, brazo abducido o brazo
    apoyado: ninguno de los tres se distingue en 2D monocular.
    """
    return np.select(
        [elevation_deg <= 20, elevation_deg <= 45, elevation_deg <= 90],
        [1, 2, 3],
        default=4,
    ).astype(int)


def lower_arm_score(flexion_deg: np.ndarray) -> np.ndarray:
    """Entre 60° y 100° 1; fuera de ese rango 2."""
    return np.where((flexion_deg >= 60) & (flexion_deg <= 100), 1, 2).astype(int)


# --- combinación ---------------------------------------------------------------


def score_a(trunk: np.ndarray, neck: np.ndarray, legs: np.ndarray) -> np.ndarray:
    columna = (np.clip(neck, 1, 3) - 1) * 4 + (np.clip(legs, 1, 4) - 1)
    return TABLE_A[np.clip(trunk, 1, 5) - 1, columna]


def score_b(arm: np.ndarray, forearm: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    columna = (np.clip(forearm, 1, 2) - 1) * 3 + (np.clip(wrist, 1, 3) - 1)
    return TABLE_B[np.clip(arm, 1, 6) - 1, columna]


def score_c(a_total: np.ndarray, b_total: np.ndarray) -> np.ndarray:
    return TABLE_C[np.clip(a_total, 1, 12) - 1, np.clip(b_total, 1, 12) - 1]


# --- ángulos a partir del esqueleto 2D ----------------------------------------


def _mid(keypoints: np.ndarray, left: str, right: str) -> np.ndarray:
    return keypoints[:, [JOINT[left], JOINT[right]], :].mean(axis=1)


def _angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    cos = np.divide((u * v).sum(axis=1), norms, out=np.zeros(len(u)), where=norms > 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def body_angles(keypoints: np.ndarray) -> dict[str, np.ndarray]:
    """Los cinco ángulos observables, en grados, por fotograma.

    En coordenadas de imagen la y crece hacia abajo, así que «arriba» es -y. El
    tronco y el cuello se miden contra la vertical; el brazo contra el tronco, que
    es lo que pide la norma y además no depende de cómo esté inclinada la persona.
    """
    hip = _mid(keypoints, "left_hip", "right_hip")
    shoulder = _mid(keypoints, "left_shoulder", "right_shoulder")
    ear = _mid(keypoints, "left_ear", "right_ear")
    arriba = np.tile([0.0, -1.0], (len(keypoints), 1))

    trunk_vec = shoulder - hip
    neck_vec = ear - shoulder

    # Se mide el lado derecho: en UW-IOM el participante trabaja de perfil y una
    # sola cámara no ve bien los dos brazos. Con varias personas o dos manos habrá
    # que decidir si se puntúa el peor lado, que es lo que haría un evaluador.
    upper_arm_vec = keypoints[:, JOINT["right_elbow"], :] - keypoints[:, JOINT["right_shoulder"], :]
    forearm_vec = keypoints[:, JOINT["right_wrist"], :] - keypoints[:, JOINT["right_elbow"], :]
    thigh_vec = keypoints[:, JOINT["right_knee"], :] - keypoints[:, JOINT["right_hip"], :]
    shin_vec = keypoints[:, JOINT["right_ankle"], :] - keypoints[:, JOINT["right_knee"], :]

    return {
        "trunk_flexion": _angle_between(trunk_vec, arriba),
        "neck_flexion": _angle_between(neck_vec, trunk_vec),
        "upper_arm_elevation": _angle_between(upper_arm_vec, -trunk_vec),
        # OJO, y aquí hubo un error que solo se vio al dibujar el esqueleto: la
        # flexión es DIRECTAMENTE el ángulo entre los dos segmentos, sin restar de
        # 180. Los vectores van hombro→codo y codo→muñeca, los dos en el mismo
        # sentido de recorrido, así que un miembro EXTENDIDO da 0 grados entre
        # ellos y uno doblado en escuadra da 90. Restarlo de 180 —que es lo que uno
        # escribe pensando en el ángulo interior de la articulación— hacía que una
        # pierna recta puntuara como flexión máxima.
        "lower_arm_flexion": _angle_between(upper_arm_vec, forearm_vec),
        "knee_flexion": _angle_between(thigh_vec, shin_vec),
    }


def reba_from_keypoints(
    keypoints: np.ndarray,
    present: np.ndarray | None = None,
    assumptions: RebaAssumptions | None = None,
    bilateral_support: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Puntaje REBA por fotograma, con todos los componentes a la vista.

    Devuelve los ángulos, los puntajes parciales y el final, para que un informe
    pueda explicar POR QUÉ una postura puntuó alto. Un número suelto no sirve: el
    valor de esto para un ingeniero de SST es saber qué articulación lo provoca.

    Donde no hay persona, el puntaje sale como 0, que no es un nivel de la norma y
    por eso no se puede confundir con un riesgo bajo.
    """
    assumptions = assumptions or RebaAssumptions()
    angles = body_angles(keypoints)
    n = len(keypoints)

    if bilateral_support is None:
        bilateral_support = np.ones(n, dtype=bool)

    trunk = trunk_score(angles["trunk_flexion"]) + int(assumptions.trunk_twist)
    neck = neck_score(angles["neck_flexion"]) + int(assumptions.neck_twist)
    legs = legs_score(angles["knee_flexion"], bilateral_support)
    arm = upper_arm_score(angles["upper_arm_elevation"])
    forearm = lower_arm_score(angles["lower_arm_flexion"])
    wrist = np.full(n, assumptions.wrist)

    a = score_a(trunk, neck, legs) + assumptions.load_force
    b = score_b(arm, forearm, wrist) + assumptions.coupling
    c = score_c(a, b)

    if present is not None:
        c = np.where(present, c, 0)

    return angles | {
        "trunk": trunk,
        "neck": neck,
        "legs": legs,
        "upper_arm": arm,
        "lower_arm": forearm,
        "wrist": wrist,
        "score_a": a,
        "score_b": b,
        "reba": c.astype(int),
    }
