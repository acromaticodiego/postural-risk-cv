"""Reconocer la tarea con REGLAS, sin aprender nada. Es el número a batir.

Si el modelo entrenado no le gana a esto, el proyecto no tiene modelo: tiene una
calculadora con pasos de más. Y saberlo el primer día cuesta una tarde; saberlo al
final cuesta el proyecto.

POR QUÉ HACE FALTA RECONOCER LA TAREA, si REBA ya da el riesgo

El puntaje dice CUÁNTO riesgo hay. La tarea dice DE QUÉ. Un informe que dice
«riesgo alto 40 minutos» no se puede accionar; uno que dice «riesgo alto 40
minutos, casi todo recogiendo de la estantería baja» se acciona subiendo la
estantería. Por eso el modelo aprende la tarea, y el riesgo se calcula aparte.

LOS UMBRALES SE CALIBRAN EN ENTRENAMIENTO, y esto tiene su motivo.

La primera versión los dejaba fijos, puestos desde la norma y el sentido común, con
el argumento de que una línea base ajustada deja de ser una línea base. Medido, el
argumento se cayó: con los umbrales a ojo las reglas sacaban 0,674 de F1 macro en
movimiento, y el riesgo de eso no es que sean demasiado fuertes sino demasiado
DÉBILES. Comparar un modelo entrenado contra un rival mal calibrado le regala una
ventaja que no tiene, y eso es tan tramposo como ajustar la vara al resultado.

Así que las reglas reciben exactamente el mismo privilegio que el modelo —mirar los
sujetos de entrenamiento de su pliegue— y ni uno más: el conjunto de prueba no se
toca. Los valores por defecto siguen siendo los de la norma, para quien quiera
correrlas sin calibrar.

QUÉ NO PUEDE SABER: si lo que se manipula es una caja o una varilla. Eso no está
en el esqueleto, y ninguna regla lo va a sacar. Se predice «none» siempre y se
publica ese fallo como lo que es: el límite de mirar solo al cuerpo. Si el modelo
entrenado acierta ahí, habrá encontrado algo que las reglas no ven —o estará
memorizando el orden del guion del experimento, que también hay que comprobar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pose.schema import JOINT

# De la norma: el tronco entra en el tramo penalizado de REBA a partir de 20°, y a
# partir de 60° es el tramo peor. 30° es el punto en que una persona ha dejado de
# estar de pie con naturalidad, y se usa para separar `bend` de `stand`.
TRUNK_BEND_DEG = 30.0

# CRITERIO, no norma: cuánto tiene que desplazarse el cuerpo para llamarlo andar.
# Medido en alturas de cuerpo por segundo, para que no dependa de la resolución ni
# de la distancia a la cámara. Un paso normal mueve el cuerpo del orden de media
# altura por segundo; se pone por debajo para no perder los desplazamientos lentos.
WALK_SPEED_BODIES_PER_SECOND = 0.25

# CRITERIO: margen, en alturas de cuerpo, para decidir que la mano está por encima
# del hombro o por debajo de la cadera en vez de justo en el límite.
HEIGHT_MARGIN = 0.05


def _mid(keypoints: np.ndarray, left: str, right: str) -> np.ndarray:
    return keypoints[:, [JOINT[left], JOINT[right]], :].mean(axis=1)


def _body_height(keypoints: np.ndarray) -> np.ndarray:
    """Altura del cuerpo en píxeles, para normalizar todo lo demás."""
    alto = keypoints[:, :, 1].max(axis=1) - keypoints[:, :, 1].min(axis=1)
    return np.maximum(alto, 1.0)


def predict_motion(
    keypoints: np.ndarray,
    trunk_flexion: np.ndarray,
    hz: float,
    trunk_bend: float = TRUNK_BEND_DEG,
    walk_speed: float = WALK_SPEED_BODIES_PER_SECOND,
) -> np.ndarray:
    """walk, stand o bend, por este orden de prioridad.

    `bend` gana a `walk` cuando coinciden: alguien que camina agachado está en la
    postura peligrosa, y es esa la que hay que contar.
    """
    caderas = _mid(keypoints, "left_hip", "right_hip")
    altura = _body_height(keypoints)
    desplazamiento = np.zeros(len(keypoints))
    desplazamiento[1:] = np.linalg.norm(np.diff(caderas, axis=0), axis=1)
    velocidad = desplazamiento / altura * hz  # alturas de cuerpo por segundo

    # Se suaviza con una ventana de medio segundo: un salto de un solo fotograma es
    # el detector titubeando, no una persona que echa a andar.
    ventana = max(1, int(round(0.5 * hz)))
    nucleo = np.ones(ventana) / ventana
    velocidad = np.convolve(velocidad, nucleo, mode="same")

    salida = np.where(velocidad > walk_speed, "walk", "stand")
    return np.where(trunk_flexion > trunk_bend, "bend", salida)


def predict_height(keypoints: np.ndarray, height_margin: float = HEIGHT_MARGIN) -> np.ndarray:
    """A qué altura trabajan las manos: low, mid o top.

    Se mira la muñeca más alta de las dos: lo que decide el riesgo del hombro es el
    brazo que más se levanta, igual que un evaluador puntúa el peor lado.
    """
    altura = _body_height(keypoints)
    hombro = _mid(keypoints, "left_shoulder", "right_shoulder")[:, 1]
    cadera = _mid(keypoints, "left_hip", "right_hip")[:, 1]
    # En coordenadas de imagen la y crece hacia abajo: la muñeca MÁS ALTA es la de
    # y más pequeña.
    muneca = np.minimum(
        keypoints[:, JOINT["left_wrist"], 1], keypoints[:, JOINT["right_wrist"], 1]
    )
    margen = height_margin * altura
    return np.where(
        muneca < hombro - margen, "top", np.where(muneca > cadera + margen, "low", "mid")
    )


def predict_manipulation(keypoints: np.ndarray, height: np.ndarray) -> np.ndarray:
    """reach, hold o none. Sin ver el objeto, esto es lo que da la postura.

    `reach` cuando el brazo está estirado lejos del cuerpo; `hold` cuando las manos
    están juntas y delante, que es como se sostiene algo con las dos manos. No se
    intenta distinguir `pick-up` de `place`: son el mismo gesto en dos sentidos y
    el esqueleto de un fotograma no sabe hacia dónde va el movimiento. Esa es
    justamente una de las cosas que un modelo temporal podría aprender y las reglas
    no, y por eso se deja fallando en vez de disimularlo.
    """
    altura_cuerpo = _body_height(keypoints)
    hombro_izq = keypoints[:, JOINT["left_shoulder"], :]
    hombro_der = keypoints[:, JOINT["right_shoulder"], :]
    muneca_izq = keypoints[:, JOINT["left_wrist"], :]
    muneca_der = keypoints[:, JOINT["right_wrist"], :]

    alcance = np.maximum(
        np.linalg.norm(muneca_izq - hombro_izq, axis=1),
        np.linalg.norm(muneca_der - hombro_der, axis=1),
    ) / altura_cuerpo
    manos_juntas = np.linalg.norm(muneca_izq - muneca_der, axis=1) / altura_cuerpo

    salida = np.full(len(keypoints), "none", dtype=object)
    salida[manos_juntas < 0.12] = "hold"
    salida[alcance > 0.38] = "reach"
    return salida.astype(str)


def predict_object(n: int) -> np.ndarray:
    """Siempre `none`: si es caja o varilla no está en el esqueleto y no se finge."""
    return np.full(n, "none")


def predict_all(
    keypoints: np.ndarray,
    trunk_flexion: np.ndarray,
    hz: float,
    thresholds: "Thresholds | None" = None,
) -> dict[str, np.ndarray]:
    t = thresholds or Thresholds()
    motion = predict_motion(keypoints, trunk_flexion, hz, t.trunk_bend, t.walk_speed)
    height = predict_height(keypoints, t.height_margin)
    return {
        "object": predict_object(len(keypoints)),
        "motion": motion,
        "manipulation": predict_manipulation(keypoints, height),
        "height": np.where(motion == "walk", "none", height),
    }


# --- calibración ---------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    """Los tres cortes de las reglas. Por defecto, los de la norma y el criterio."""

    trunk_bend: float = TRUNK_BEND_DEG
    walk_speed: float = WALK_SPEED_BODIES_PER_SECOND
    height_margin: float = HEIGHT_MARGIN


def calibrate(
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]], hz: float
) -> Thresholds:
    """Busca los umbrales que mejor van EN ENTRENAMIENTO, y solo ahí.

    Esto no es hacer trampa, es lo contrario. Una línea base con umbrales puestos a
    ojo puede salir artificialmente débil, y comparar un modelo entrenado contra un
    rival mal calibrado le regala una ventaja que no tiene. Se le concede a las
    reglas exactamente el mismo privilegio que al modelo —mirar el entrenamiento— y
    ni uno más: el conjunto de prueba de cada pliegue no se toca.

    `samples` son tríos (keypoints, flexión del tronco, etiquetas) de los sujetos
    de entrenamiento del pliegue.
    """
    from ..eval.metrics import report

    keypoints = np.concatenate([s[0] for s in samples])
    flexion = np.concatenate([s[1] for s in samples])
    etiquetas = np.concatenate([s[2] for s in samples])
    motion_true, height_true = etiquetas[:, 1], etiquetas[:, 3]

    mejor_tronco, mejor_velocidad, mejor_motion = TRUNK_BEND_DEG, WALK_SPEED_BODIES_PER_SECOND, -1.0
    for tronco in np.arange(10.0, 55.1, 5.0):
        for velocidad in np.arange(0.05, 0.61, 0.05):
            puntaje = report(
                motion_true, predict_motion(keypoints, flexion, hz, tronco, velocidad)
            ).macro_f1
            if puntaje > mejor_motion:
                mejor_tronco, mejor_velocidad, mejor_motion = tronco, velocidad, puntaje

    motion = predict_motion(keypoints, flexion, hz, mejor_tronco, mejor_velocidad)
    mejor_margen, mejor_altura = HEIGHT_MARGIN, -1.0
    for margen in np.arange(0.0, 0.201, 0.02):
        predicho = np.where(motion == "walk", "none", predict_height(keypoints, margen))
        puntaje = report(height_true, predicho).macro_f1
        if puntaje > mejor_altura:
            mejor_margen, mejor_altura = margen, puntaje

    return Thresholds(
        trunk_bend=float(mejor_tronco),
        walk_speed=float(mejor_velocidad),
        height_margin=float(mejor_margen),
    )
