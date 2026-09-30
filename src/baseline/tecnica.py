"""¿El riesgo que marca el sistema es evitable o es el propio de la tarea?

NACE DE UNA OBSERVACIÓN DE JUAN DIEGO (30/09) grabándose a sí mismo: «de lado
siempre me pone naranja sin importar cómo lo haga». Tenía razón, y los datos de su
propio vídeo lo confirman:

| cómo se agacha            | tronco | piernas | REBA |
|---------------------------|--------|---------|------|
| doblando la espalda       |   4    |    1    |  4   |
| en cuclillas, espalda recta (16°) | 2 | 3    |  4   |

**REBA compensa exactamente.** Lo que se gana en el tronco al agacharse bien se
pierde en las piernas al flexionar las rodillas, y el puntaje final no se mueve.

No es un fallo del cálculo: la norma es así, porque puntúa la carga articular total
y unas rodillas muy flexionadas también cargan. Pero como producto es un problema
grave: **un operario que trabaja bien ve el mismo naranja que uno que trabaja mal**,
y entonces deja de mirar el panel. Un sistema de seguridad al que se deja de hacer
caso vale exactamente cero.

LA SOLUCIÓN: separar el riesgo EVITABLE del INHERENTE a la tarea. Recoger algo del
suelo tiene un mínimo que no se puede bajar —hay que llegar abajo de alguna manera—
y encima de ese mínimo está lo que sí depende de cómo se haga. Lo que el operario
necesita oír no es «naranja», es «esto ya es lo mejor que se puede hacer» o «dobla
las rodillas».
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Un tronco por encima de este puntaje está flexionado de más: 3 significa más de
# 20 grados, que es donde la norma empieza a penalizar de verdad.
TRUNK_BENT = 3

# Unas piernas en 3 o más significan rodilla flexionada por encima de 30 grados: la
# persona ha bajado el cuerpo en vez de doblar la espalda.
LEGS_SQUATTING = 3


@dataclass(frozen=True)
class TechniqueVerdict:
    """Qué parte del riesgo depende de cómo se hace el trabajo."""

    code: str
    message: str
    avoidable: bool
    """True si hay margen real de mejora cambiando la técnica. False cuando lo que
    marca es el mínimo de la tarea, y entonces lo que hay que rediseñar es el puesto
    —subir la estantería, acercar la carga— y no regañar a nadie."""


VERDICTS = {
    "espalda": TechniqueVerdict(
        "espalda",
        "Estás doblando la espalda con las piernas rectas. Dobla las rodillas y baja el cuerpo.",
        avoidable=True,
    ),
    "cuclillas": TechniqueVerdict(
        "cuclillas",
        "Técnica correcta: espalda recta y rodillas dobladas. El riesgo que queda es "
        "el propio de trabajar a esta altura, y se baja subiendo la carga, no cambiando la postura.",
        avoidable=False,
    ),
    "forzada": TechniqueVerdict(
        "forzada",
        "Espalda doblada Y rodillas flexionadas a la vez: es la postura que más carga "
        "la columna. Acércate a la carga antes de bajar.",
        avoidable=True,
    ),
    "brazos": TechniqueVerdict(
        "brazos",
        "El riesgo viene de los brazos, no de la espalda. Acerca la carga al cuerpo "
        "o baja la altura a la que trabajas.",
        avoidable=True,
    ),
    "neutra": TechniqueVerdict(
        "neutra", "Postura sin riesgo apreciable.", avoidable=False
    ),
}


def assess_technique(
    trunk: np.ndarray, legs: np.ndarray, upper_arm: np.ndarray, reba: np.ndarray, threshold: int = 4
) -> np.ndarray:
    """Un veredicto de técnica por fotograma.

    El orden de las comprobaciones importa y es el de la gravedad: la postura
    forzada —espalda doblada y rodillas flexionadas a la vez— es la peor y se mira
    primero, porque si no quedaría clasificada como «espalda» y el consejo sería
    incompleto.

    Por debajo del umbral de acción no se emite consejo. Decirle a alguien cómo
    agacharse cuando no está en riesgo es la forma más rápida de que deje de leer
    los avisos.
    """
    trunk, legs = np.asarray(trunk), np.asarray(legs)
    upper_arm, reba = np.asarray(upper_arm), np.asarray(reba)

    doblado = trunk >= TRUNK_BENT
    cuclillas = legs >= LEGS_SQUATTING
    brazos_altos = upper_arm >= 3

    codigos = np.where(
        reba < threshold,
        "neutra",
        np.where(
            doblado & cuclillas,
            "forzada",
            np.where(
                doblado,
                "espalda",
                np.where(cuclillas, "cuclillas", np.where(brazos_altos, "brazos", "neutra")),
            ),
        ),
    )
    return codigos


def avoidable_share(codes: np.ndarray, reba: np.ndarray, threshold: int = 4) -> dict:
    """Cuánto del tiempo en riesgo es evitable cambiando la técnica.

    Es la cifra que convierte el informe en dos acciones distintas: la parte
    evitable se corrige formando a la gente, y la inherente **solo** se corrige
    rediseñando el puesto. Mezclarlas lleva a dar formación donde hacía falta una
    mesa más alta.
    """
    en_riesgo = np.asarray(reba) >= threshold
    if not en_riesgo.any():
        return {"frames_at_risk": 0, "avoidable": 0, "inherent": 0, "avoidable_share": 0.0}
    codigos = np.asarray(codes)[en_riesgo]
    evitables = sum(VERDICTS[c].avoidable for c in codigos)
    return {
        "frames_at_risk": int(en_riesgo.sum()),
        "avoidable": int(evitables),
        "inherent": int(en_riesgo.sum() - evitables),
        "avoidable_share": round(float(evitables / en_riesgo.sum()), 3),
    }
