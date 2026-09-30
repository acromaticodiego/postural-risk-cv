"""Que el consejo de técnica distinga lo evitable de lo inherente a la tarea.

La prueba que da sentido al módulo es
`test_the_two_ways_of_bending_get_the_same_reba_but_different_advice`: con datos
reales del vídeo del 30/09, agacharse doblando la espalda y agacharse en cuclillas
dan **el mismo REBA de 4**, y uno es evitable y el otro es el mínimo de la tarea.
Si el consejo no los separase, el módulo no aportaría nada sobre el color.

    .\\.venv\\Scripts\\python.exe -m tests.test_tecnica
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline.tecnica import VERDICTS, assess_technique, avoidable_share  # noqa: E402


def _codigos(trunk, legs, upper_arm, reba):
    return assess_technique(
        np.array(trunk), np.array(legs), np.array(upper_arm), np.array(reba)
    )


def test_the_two_ways_of_bending_get_the_same_reba_but_different_advice() -> None:
    """El caso que motivó el módulo, con los puntajes medidos en el vídeo real."""
    # Doblando la espalda: tronco 4, piernas 1, REBA 4.
    # En cuclillas con la espalda recta: tronco 2, piernas 3, REBA 4.
    codigos = _codigos(trunk=[4, 2], legs=[1, 3], upper_arm=[1, 1], reba=[4, 4])
    assert codigos[0] == "espalda" and codigos[1] == "cuclillas"
    assert VERDICTS[codigos[0]].avoidable is True, "doblar la espalda sí se puede evitar"
    assert VERDICTS[codigos[1]].avoidable is False, (
        "las cuclillas con la espalda recta son el mínimo de agacharse: no hay nada "
        "que corregir en la persona, hay que subir la carga"
    )


def test_the_worst_posture_is_detected_before_the_others() -> None:
    """Espalda doblada Y rodillas flexionadas a la vez es lo peor, y se mira primero.

    Sin ese orden quedaría clasificada como «espalda» y el consejo sería incompleto.
    """
    assert _codigos([4], [3], [1], [7])[0] == "forzada"


def test_no_advice_below_the_action_threshold() -> None:
    """Decirle a alguien cómo agacharse cuando no está en riesgo es la forma más
    rápida de que deje de leer los avisos."""
    codigos = _codigos(trunk=[4, 4], legs=[1, 1], upper_arm=[1, 1], reba=[3, 4])
    assert codigos[0] == "neutra" and codigos[1] == "espalda"


def test_arms_are_named_when_they_are_the_cause() -> None:
    """Si el riesgo viene del brazo, el consejo no puede hablar de la espalda."""
    codigo = _codigos(trunk=[1], legs=[1], upper_arm=[4], reba=[6])[0]
    assert codigo == "brazos"
    assert "brazos" in VERDICTS[codigo].message or "carga" in VERDICTS[codigo].message


def test_every_verdict_has_a_message_and_a_decision() -> None:
    for codigo, v in VERDICTS.items():
        assert v.message and len(v.message) > 20, f"{codigo} sin mensaje util"
        assert isinstance(v.avoidable, bool)


def test_avoidable_share_separates_the_two_kinds_of_action() -> None:
    """La cifra que decide si hay que formar a la gente o rediseñar el puesto.

    Mezclarlas lleva a dar formación donde hacía falta una mesa más alta.
    """
    codigos = np.array(["espalda", "espalda", "cuclillas", "neutra"])
    reba = np.array([5, 6, 4, 2])
    r = avoidable_share(codigos, reba)
    assert r["frames_at_risk"] == 3
    assert r["avoidable"] == 2 and r["inherent"] == 1
    # La fracción se publica redondeada a tres decimales, así que la tolerancia
    # tiene que admitirlo: compararla con 2/3 exacto y 1e-9 falla por el redondeo,
    # no por el cálculo.
    assert abs(r["avoidable_share"] - 2 / 3) < 1e-3


def test_no_risk_gives_no_shares() -> None:
    r = avoidable_share(np.array(["neutra"] * 4), np.array([1, 2, 3, 1]))
    assert r["frames_at_risk"] == 0 and r["avoidable_share"] == 0.0


if __name__ == "__main__":
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fallos = 0
    for prueba in pruebas:
        try:
            prueba()
            print(f"  ok   {prueba.__name__}")
        except AssertionError as e:
            fallos += 1
            print(f"  FALLA {prueba.__name__}: {e}")
    print(f"\n{len(pruebas) - fallos}/{len(pruebas)}")
    sys.exit(1 if fallos else 0)
