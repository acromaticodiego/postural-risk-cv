"""La ecuación NIOSH de levantamiento, que es lo que REBA no sabe medir.

POR QUÉ HACE FALTA. La primera versión de este módulo decía que REBA no distingue
las dos técnicas de levantar —espalda doblada contra rodillas dobladas— y NIOSH sí.
**Medido, era falso**: al alejar la carga del cuerpo también se inclina el tronco y
se eleva el brazo, así que REBA sube igual (de 3 a 6 en el mismo caso). La
afirmación venía de generalizar un ejemplo concreto, con las manos en el mismo
sitio, donde REBA se movía un solo punto. Queda como la medición falsa nº 4.

Lo que NIOSH sí aporta, comprobado en `tests/test_niosh.py`:

  · **Un resultado en kilogramos.** Cuánto debería pesar la carga para que ese
    levantamiento fuera aceptable. Un puntaje ordinal de 6 no le dice a nadie qué
    hacer; «esta caja de 12 kg debería pesar 8» se lleva a una reunión.
  · **Un umbral con el peso real dentro.** El índice cruza 1 cuando el
    levantamiento deja de ser recomendable. REBA solo puede meter el peso como un
    ajuste grosero de 0 a 3.
  · **Frecuencia y recorrido**, que REBA no modela.

QUÉ SE MIDE CON LA CÁMARA Y QUÉ SE DECLARA

  H, la distancia horizontal de las manos al cuerpo   ← la cámara, y es la clave
  V, la altura de las manos sobre el suelo            ← la cámara
  D, el recorrido vertical de la carga                ← la cámara, con la secuencia
  F, la frecuencia de levantamientos                  ← la cámara, con la secuencia
  A, el ángulo de asimetría (torsión)                 ← NO en 2D: se declara
  el peso de la carga                                 ← se declara (ADR 0001)
  C, la calidad del agarre                            ← se declara (ADR 0001)

EL PROBLEMA DE LA ESCALA, y cómo se resuelve. La ecuación trabaja en centímetros y
la cámara da píxeles. La salida es la misma del ADR 0001: el puesto declara la
estatura del trabajador y el cuerpo hace de regla, que es literalmente lo que hace
un evaluador midiendo sobre una fotografía. Sin estatura declarada se asume 170 cm
y el resultado se marca como estimado, porque un número en centímetros sacado de
una estatura inventada tiene que decir que lo es.

Fuente: Waters, Putz-Anderson y Garg, *Applications Manual for the Revised NIOSH
Lifting Equation* (1994). Los multiplicadores y sus cortes son de ahí.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pose.schema import JOINT

LOAD_CONSTANT_KG = 23.0
DEFAULT_WORKER_HEIGHT_CM = 170.0

# Multiplicador de frecuencia, tabla de la norma para trabajo de hasta 1 hora.
# Se simplifica a la fila de V >= 75 cm y se documenta: la fila de V < 75 castiga
# más, así que esta elección es la OPTIMISTA, y un sistema que estima el riesgo
# tiene que decir hacia dónde se equivoca.
FREQUENCY_MULTIPLIER = {
    0.2: 1.00, 0.5: 0.97, 1: 0.94, 2: 0.91, 3: 0.88, 4: 0.84, 5: 0.80,
    6: 0.75, 7: 0.70, 8: 0.60, 9: 0.52, 10: 0.45, 11: 0.41, 12: 0.37,
    13: 0.34, 14: 0.31, 15: 0.28,
}

COUPLING_MULTIPLIER = {"good": 1.00, "fair": 0.95, "poor": 0.90, "unacceptable": 0.90}


@dataclass(frozen=True)
class LiftAnalysis:
    """Un levantamiento analizado con la ecuación NIOSH."""

    horizontal_cm: float
    vertical_cm: float
    travel_cm: float
    frequency_per_min: float
    asymmetry_deg: float
    multipliers: dict[str, float]
    recommended_weight_kg: float
    """RWL: cuánto peso sería aceptable levantar ASÍ. Baja al empeorar la técnica."""
    lifting_index: float | None
    """Peso real dividido por el recomendado. Por encima de 1 hay riesgo. None si
    el puesto no declaró el peso."""
    scale_estimated: bool
    """True si la estatura del trabajador no se declaró y se asumió una media."""

    measured_at_load: bool = False
    """True si la distancia horizontal se midió al CENTRO DE LA CARGA detectada, que
    es lo que pide la norma. False si se usó el punto medio de las muñecas, que la
    subestima un poco porque el centro de una caja queda por delante de los
    nudillos."""

    @property
    def verdict(self) -> str:
        if self.lifting_index is None:
            return "sin peso declarado: no se puede dar el índice"
        if self.lifting_index <= 1.0:
            return "dentro de lo recomendado"
        if self.lifting_index <= 2.0:
            return "por encima de lo recomendado"
        if self.lifting_index <= 3.0:
            return "riesgo alto"
        return "riesgo muy alto"

    @property
    def worst_factor(self) -> str:
        """El multiplicador que más está penalizando. Es lo que hay que corregir."""
        return min(self.multipliers, key=self.multipliers.get)


# --- los multiplicadores de la norma -----------------------------------------


def horizontal_multiplier(h_cm: float) -> float:
    """25/H. Es el factor que más castiga: alejar la carga del cuerpo la penaliza
    proporcionalmente, y por encima de 63 cm la norma da el levantamiento por
    inaceptable."""
    if h_cm > 63.0:
        return 0.0
    return 1.0 if h_cm <= 25.0 else 25.0 / h_cm


def vertical_multiplier(v_cm: float) -> float:
    """Óptimo a la altura de los nudillos (75 cm). Penaliza por arriba y por abajo."""
    if v_cm > 175.0:
        return 0.0
    return max(0.0, 1.0 - 0.003 * abs(v_cm - 75.0))


def distance_multiplier(d_cm: float) -> float:
    """Cuánto sube la carga. Por debajo de 25 cm no penaliza."""
    if d_cm > 175.0:
        return 0.0
    return 1.0 if d_cm < 25.0 else 0.82 + 4.5 / d_cm


def asymmetry_multiplier(a_deg: float) -> float:
    """La torsión del tronco. En 2D no es observable, así que entra por configuración."""
    return 0.0 if a_deg > 135.0 else max(0.0, 1.0 - 0.0032 * a_deg)


def frequency_multiplier(lifts_per_min: float) -> float:
    claves = sorted(FREQUENCY_MULTIPLIER)
    if lifts_per_min > claves[-1]:
        return 0.0
    for clave in claves:
        if lifts_per_min <= clave:
            return FREQUENCY_MULTIPLIER[clave]
    return 0.0


# --- de esqueleto a centímetros ----------------------------------------------


def _pixels_per_cm(keypoints: np.ndarray, worker_height_cm: float) -> float:
    """La escala, usando el cuerpo como regla.

    Se mide de la coronilla al tobillo, no la caja entera: la caja crece cuando la
    persona extiende un brazo, y entonces la misma persona daría dos escalas
    distintas en dos fotogramas seguidos.
    """
    tobillos = keypoints[[JOINT["left_ankle"], JOINT["right_ankle"]], 1]
    suelo = float(np.max(tobillos))
    cabeza = float(np.min(keypoints[keypoints[:, 1] > 0, 1])) if (keypoints[:, 1] > 0).any() else 0.0
    alto_px = max(suelo - cabeza, 1.0)
    # La coronilla queda por encima de la nariz y los ojos, que es lo más alto que
    # da el esqueleto: se corrige con la proporción anatómica habitual.
    return alto_px / (worker_height_cm * 0.94)


def analyze_lift(
    keypoints_sequence: np.ndarray,
    hz: float,
    worker_height_cm: float | None = None,
    load_kg: float | None = None,
    coupling: str | None = None,
    asymmetry_deg: float = 0.0,
    lifts_per_min: float = 1.0,
    load_centers: np.ndarray | None = None,
) -> LiftAnalysis:
    """Analiza un levantamiento a partir de la secuencia de esqueletos del evento.

    Los multiplicadores se calculan en el ORIGEN del levantamiento —el fotograma en
    que la carga está más baja—, que es lo que pide la norma: es el instante en que
    la columna soporta el peor momento de fuerza.

    `load_centers` son los centros de la carga detectada, uno por fotograma. Cuando
    se dan, la distancia horizontal se mide **al centro de la carga**, que es lo que
    la norma pide literalmente. Sin ellos se usa el punto medio de las muñecas, que
    es una aproximación razonable —la carga está donde están las manos— y que
    subestima un poco: el centro de una caja queda por delante de los nudillos, así
    que la distancia real es algo mayor que la medida. Conviene saber hacia dónde se
    equivoca cada versión.
    """
    estimada = worker_height_cm is None
    altura = worker_height_cm or DEFAULT_WORKER_HEIGHT_CM

    if load_centers is not None and len(load_centers) == len(keypoints_sequence):
        manos = np.asarray(load_centers, dtype=np.float64)
    else:
        manos = keypoints_sequence[
            :, [JOINT["left_wrist"], JOINT["right_wrist"]], :
        ].mean(axis=1)
    # En coordenadas de imagen la y crece hacia abajo: las manos MÁS BAJAS son las
    # de y mayor.
    origen = int(np.argmax(manos[:, 1]))
    destino = int(np.argmin(manos[:, 1]))

    esqueleto = keypoints_sequence[origen]
    px_cm = _pixels_per_cm(esqueleto, altura)

    tobillos = esqueleto[[JOINT["left_ankle"], JOINT["right_ankle"]], :]
    suelo_y = float(np.max(tobillos[:, 1]))
    centro_pies_x = float(np.mean(tobillos[:, 0]))

    h_cm = abs(float(manos[origen, 0]) - centro_pies_x) / px_cm
    v_cm = max(0.0, (suelo_y - float(manos[origen, 1])) / px_cm)
    d_cm = abs(float(manos[destino, 1]) - float(manos[origen, 1])) / px_cm

    multiplicadores = {
        "horizontal": horizontal_multiplier(h_cm),
        "vertical": vertical_multiplier(v_cm),
        "recorrido": distance_multiplier(d_cm),
        "asimetria": asymmetry_multiplier(asymmetry_deg),
        "frecuencia": frequency_multiplier(lifts_per_min),
        "agarre": COUPLING_MULTIPLIER.get(coupling or "good", 1.0),
    }
    rwl = LOAD_CONSTANT_KG * float(np.prod(list(multiplicadores.values())))

    return LiftAnalysis(
        horizontal_cm=round(h_cm, 1),
        vertical_cm=round(v_cm, 1),
        travel_cm=round(d_cm, 1),
        frequency_per_min=lifts_per_min,
        asymmetry_deg=asymmetry_deg,
        multipliers={k: round(v, 3) for k, v in multiplicadores.items()},
        recommended_weight_kg=round(rwl, 2),
        lifting_index=round(load_kg / rwl, 2) if load_kg and rwl > 0 else None,
        scale_estimated=estimada,
        measured_at_load=load_centers is not None and len(load_centers) == len(keypoints_sequence),
    )
