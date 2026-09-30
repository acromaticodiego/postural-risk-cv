"""Que el puesto de trabajo cumpla el ADR 0001, sobre todo sus dos caras.

La prueba que más protege es `test_short_risk_counts_as_exposure_but_not_as_event`:
el ADR decide que la exposición acumulada NO filtra por duración y que los eventos
SÍ, y son dos caras que no se mezclan. Si alguien las unificara —cualquiera de las
dos direcciones parece una simplificación razonable— el informe se sesgaría a la
baja o las alertas se llenarían de titubeos del detector.

    .\\.venv\\Scripts\\python.exe -m tests.test_workstation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.product.workstation import (  # noqa: E402
    COMPONENTS,
    WorkstationConfig,
    summarize_exposure,
)

HZ = 10.0


def _scores(reba: list[int], dominante: str = "trunk") -> dict[str, np.ndarray]:
    """Puntajes sintéticos por fotograma, con un componente claramente dominante."""
    n = len(reba)
    s = {nombre: np.ones(n) for nombre in COMPONENTS}
    s[dominante] = np.full(n, 4.0)
    s["reba"] = np.array(reba, dtype=int)
    return s


# --- la decisión central del ADR ---------------------------------------------


def test_short_risk_counts_as_exposure_but_not_as_event() -> None:
    """Medio segundo de riesgo: cero eventos, pero medio segundo de exposición.

    Las posturas breves y repetidas son el mecanismo de la lesión por esfuerzo
    repetitivo, así que filtrarlas del informe lo sesgaría justo donde importa.
    """
    reba = [1] * 10 + [6] * 5 + [1] * 10  # 0,5 s de riesgo medio a 10 Hz
    resumen = summarize_exposure(_scores(reba), HZ, WorkstationConfig("L3", "Línea 3"))

    assert resumen.events == [], f"medio segundo no debería abrir evento: {resumen.events}"
    assert abs(resumen.seconds_by_level["medio"] - 0.5) < 1e-6, (
        f"la exposición sí tiene que contarlo: {resumen.seconds_by_level}"
    )
    assert abs(resumen.seconds_at_risk - 0.5) < 1e-6


def test_sustained_risk_opens_one_event_with_its_peak_and_cause() -> None:
    reba = [1] * 5 + [6] * 12 + [9] * 8 + [1] * 5  # 2,0 s seguidos de riesgo
    resumen = summarize_exposure(
        _scores(reba, dominante="upper_arm"), HZ, WorkstationConfig("L3", "Línea 3")
    )

    assert len(resumen.events) == 1, f"debería haber un evento: {resumen.events}"
    evento = resumen.events[0]
    assert abs(evento.duration_seconds - 2.0) < 1e-6, evento.duration_seconds
    assert evento.peak_reba == 9 and evento.peak_level == "alto"
    assert evento.dominant_component == "upper_arm", (
        f"el informe tiene que decir qué articulación lo causó, dijo {evento.dominant_component}"
    )
    assert abs(evento.start_seconds - 0.5) < 1e-6


def test_two_separate_events_are_not_merged() -> None:
    reba = [1] * 3 + [6] * 12 + [1] * 6 + [6] * 12 + [1] * 3
    resumen = summarize_exposure(_scores(reba), HZ, WorkstationConfig("L3", "Línea 3"))
    assert len(resumen.events) == 2, f"dos tramos separados, no uno: {resumen.events}"


def test_event_at_the_very_end_is_not_lost() -> None:
    """Un tramo que llega hasta el último fotograma. Es el fallo de bucle clásico."""
    reba = [1] * 5 + [7] * 15
    resumen = summarize_exposure(_scores(reba), HZ, WorkstationConfig("L3", "Línea 3"))
    assert len(resumen.events) == 1, "se perdió el evento que termina con la secuencia"
    assert abs(resumen.events[0].duration_seconds - 1.5) < 1e-6


# --- la configuración del puesto ---------------------------------------------


def test_load_bands_follow_the_standard() -> None:
    assert WorkstationConfig("a", "a", load_kg=3).load_score() == 0
    assert WorkstationConfig("a", "a", load_kg=7).load_score() == 1
    assert WorkstationConfig("a", "a", load_kg=25).load_score() == 2
    assert WorkstationConfig("a", "a", load_kg=25, sudden_load=True).load_score() == 3
    # Sin declarar carga no se penaliza: eso es lo que hace el puntaje una cota inferior.
    assert WorkstationConfig("a", "a").load_score() == 0


def test_unconfigured_workstation_is_flagged_as_lower_bound() -> None:
    """Es lo que impide leer «sin alertas» como «sin riesgo»."""
    sin_configurar = WorkstationConfig("L3", "Línea 3")
    configurado = WorkstationConfig("L3", "Línea 3", load_kg=12, coupling="fair")
    assert not sin_configurar.is_configured
    assert configurado.is_configured

    reba = [1] * 5 + [6] * 12
    assert summarize_exposure(_scores(reba), HZ, sin_configurar).is_lower_bound
    assert not summarize_exposure(_scores(reba), HZ, configurado).is_lower_bound


def test_configuring_the_load_can_only_raise_the_reba_score() -> None:
    """Declarar una carga pesada nunca puede bajar el riesgo del puesto."""
    from src.baseline.reba import reba_from_keypoints
    from tests.test_reba import _bent_over

    kp = _bent_over()
    neutro = int(reba_from_keypoints(kp, assumptions=WorkstationConfig("a", "a").assumptions())["reba"][0])
    for kg in (3, 7, 25):
        cfg = WorkstationConfig("a", "a", load_kg=kg, coupling="poor", task_requires_twist=True)
        cargado = int(reba_from_keypoints(kp, assumptions=cfg.assumptions())["reba"][0])
        assert cargado >= neutro, f"con {kg} kg bajó de {neutro} a {cargado}"


def test_invalid_config_is_rejected() -> None:
    for kwargs in (
        {"coupling": "regular"},
        {"load_kg": -1},
        {"risk_threshold": 0},
        {"risk_threshold": 20},
    ):
        try:
            WorkstationConfig("a", "a", **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"aceptó una configuración imposible: {kwargs}")


def test_threshold_is_a_parameter_not_a_constant() -> None:
    """Un cliente con otro criterio lo cambia, y el resumen dice con qué valor salió."""
    reba = [3] * 20
    laxo = summarize_exposure(_scores(reba), HZ, WorkstationConfig("a", "a"))
    estricto = summarize_exposure(_scores(reba), HZ, WorkstationConfig("a", "a", risk_threshold=3))
    assert laxo.events == [] and len(estricto.events) == 1
    assert laxo.threshold == 4 and estricto.threshold == 3


def test_time_adds_up() -> None:
    """La suma de todos los niveles más los huecos tiene que ser el tiempo medido."""
    reba = [0, 0] + [1] * 4 + [3] * 4 + [6] * 6 + [12] * 4
    present = np.array([False, False] + [True] * 18)
    resumen = summarize_exposure(_scores(reba), HZ, WorkstationConfig("a", "a"), present=present)
    total = sum(resumen.seconds_by_level.values())
    assert abs(total - resumen.measured_seconds) < 1e-6, (
        f"{total} contabilizado contra {resumen.measured_seconds} medido"
    )
    assert abs(resumen.seconds_without_person - 0.2) < 1e-6


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
