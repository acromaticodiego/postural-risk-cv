"""Que el informe por tareas ordene bien, porque de ese orden cuelga una inversión.

La prueba que más protege es `test_long_moderate_task_outranks_short_severe_one`:
el informe tiene que ordenar por riesgo ACUMULADO y no por severidad máxima. Con
datos reales salió que la tarea con el REBA más alto (9, agacharse al suelo) aporta
el 9% del riesgo, mientras una de REBA 4 que dura cinco veces más aporta el 26%.
Si el orden se invirtiera, el informe mandaría a rediseñar el puesto equivocado.

    .\\.venv\\Scripts\\python.exe -m tests.test_report
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.product.report import build_report, ranking_agreement  # noqa: E402
from src.product.workstation import COMPONENTS, WorkstationConfig  # noqa: E402

HZ = 10.0
CONFIG = WorkstationConfig("L3", "Linea 3")


def _scores(reba: list[int], dominante: str = "trunk") -> dict[str, np.ndarray]:
    n = len(reba)
    s = {c: np.ones(n) for c in COMPONENTS}
    s[dominante] = np.full(n, 4.0)
    s["reba"] = np.array(reba, dtype=int)
    return s


def test_long_moderate_task_outranks_short_severe_one() -> None:
    """Lo que manda es el riesgo acumulado, no el pico."""
    # 30 s de riesgo moderado (REBA 5) contra 6 s de riesgo muy alto (11). Las dos
    # duran más que el suelo de `min_seconds`, para que la comparación sea del
    # criterio de orden y no del filtro.
    tareas = np.array(["moderada larga"] * 300 + ["severa corta"] * 60)
    informe = build_report(tareas, _scores([5] * 300 + [11] * 60), HZ, CONFIG)

    nombres = [t.task for t in informe.tasks]
    assert set(nombres) == {"moderada larga", "severa corta"}, (
        f"alguna tarea se quedó fuera del informe y la comparación no valdría: {nombres}"
    )
    assert informe.top_task.task == "moderada larga", (
        f"ordenó por severidad en vez de por riesgo acumulado: {informe.top_task.task}"
    )
    fracciones = {t.task: t.share_of_risk for t in informe.tasks}
    assert abs(fracciones["moderada larga"] - 300 / 360) < 1e-6
    assert abs(fracciones["severa corta"] - 60 / 360) < 1e-6
    # Y la severa sigue teniendo el REBA más alto: el informe no la esconde, la
    # ordena por debajo. Confundir las dos cosas sería el error contrario.
    severa = next(t for t in informe.tasks if t.task == "severa corta")
    assert severa.median_reba > informe.top_task.median_reba


def test_shares_add_up_to_one() -> None:
    tareas = np.array(["a"] * 60 + ["b"] * 80 + ["c"] * 100)
    informe = build_report(tareas, _scores([6] * 60 + [2] * 80 + [9] * 100), HZ, CONFIG)
    assert abs(sum(t.share_of_risk for t in informe.tasks) - 1.0) < 1e-6


def test_task_below_the_floor_is_left_out_but_still_counted_in_the_total() -> None:
    """Una tarea de tres segundos da un porcentaje inestable que puede encabezar el
    ranking por azar, así que se oculta; pero su tiempo no desaparece del total."""
    tareas = np.array(["habitual"] * 200 + ["fugaz"] * 10)
    informe = build_report(tareas, _scores([6] * 210), HZ, CONFIG, min_seconds=5.0)
    nombres = [t.task for t in informe.tasks]
    assert "fugaz" not in nombres and "habitual" in nombres
    assert abs(informe.total_seconds - 21.0) < 1e-6, informe.total_seconds
    assert abs(informe.total_seconds_at_risk - 21.0) < 1e-6


def test_no_risk_gives_an_honest_recommendation() -> None:
    tareas = np.array(["tranquila"] * 100)
    informe = build_report(tareas, _scores([2] * 100), HZ, CONFIG)
    assert informe.total_seconds_at_risk == 0.0
    assert "No se detect" in informe.recommendation()


def test_unconfigured_workstation_warns_in_the_recommendation() -> None:
    """La frase que impide leer «poco riesgo» como «puesto seguro»."""
    tareas = np.array(["cargar"] * 100)
    scores = _scores([7] * 100)
    sin_configurar = build_report(tareas, scores, HZ, WorkstationConfig("L3", "L3"))
    configurado = build_report(
        tareas, scores, HZ, WorkstationConfig("L3", "L3", load_kg=12, coupling="fair")
    )
    assert "mayor que el reportado" in sin_configurar.recommendation()
    assert "mayor que el reportado" not in configurado.recommendation()


def test_recommendation_names_the_guilty_joint() -> None:
    """Sin la articulación, el informe no dice qué rediseñar."""
    tareas = np.array(["alcanzar alto"] * 100)
    informe = build_report(tareas, _scores([7] * 100, dominante="upper_arm"), HZ, CONFIG)
    assert "brazo" in informe.recommendation(), informe.recommendation()


def test_ranking_agreement_detects_a_changed_priority() -> None:
    """Si el modelo cambia la tarea prioritaria, esto tiene que verse.

    Es la comprobación de la comprobación: si `ranking_agreement` dijera siempre que
    sí, la pregunta de producto del proyecto no estaría midiendo nada.
    """
    scores = _scores([6] * 120)
    verdad = build_report(np.array(["A"] * 80 + ["B"] * 40), scores, HZ, CONFIG)
    igual = build_report(np.array(["A"] * 80 + ["B"] * 40), scores, HZ, CONFIG)
    distinto = build_report(np.array(["B"] * 80 + ["A"] * 40), scores, HZ, CONFIG)

    assert ranking_agreement(verdad, igual)["top1_agree"] is True
    assert ranking_agreement(verdad, distinto)["top1_agree"] is False


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
