"""Que el cálculo de REBA respete la norma, y que las tablas no estén mal copiadas.

La prueba que más protege es la de MONOTONÍA. Las tablas de REBA tienen que crecer
—o al menos no decrecer— en cada uno de sus ejes: empeorar la postura de una
articulación no puede bajar el puntaje. Es una propiedad de la norma, no una
suposición, y sirve de detector de erratas: una celda mal transcrita rompe la
monotonía en su fila o su columna aunque el valor parezca plausible.

    .\\.venv\\Scripts\\python.exe -m tests.test_reba
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline.reba import (  # noqa: E402
    body_angles,
    TABLE_A,
    TABLE_B,
    TABLE_C,
    RebaAssumptions,
    action_level,
    lower_arm_score,
    neck_score,
    reba_from_keypoints,
    score_a,
    score_b,
    trunk_score,
    upper_arm_score,
)
from src.pose.schema import JOINT, N_JOINTS  # noqa: E402


# --- las tablas ---------------------------------------------------------------


def test_table_shapes_and_ranges() -> None:
    assert TABLE_A.shape == (5, 12), f"tabla A: {TABLE_A.shape}"
    assert TABLE_B.shape == (6, 6), f"tabla B: {TABLE_B.shape}"
    assert TABLE_C.shape == (12, 12), f"tabla C: {TABLE_C.shape}"
    assert TABLE_A.min() >= 1 and TABLE_A.max() <= 9
    assert TABLE_B.min() >= 1 and TABLE_B.max() <= 9
    assert TABLE_C.min() >= 1 and TABLE_C.max() <= 12


def test_table_c_is_monotonic() -> None:
    """En los dos ejes: un grupo A o B peor no puede dar un REBA menor."""
    fallos = []
    for fila in range(TABLE_C.shape[0]):
        diffs = np.diff(TABLE_C[fila])
        if (diffs < 0).any():
            fallos.append(f"fila A={fila + 1} baja: {TABLE_C[fila].tolist()}")
    for columna in range(TABLE_C.shape[1]):
        diffs = np.diff(TABLE_C[:, columna])
        if (diffs < 0).any():
            fallos.append(f"columna B={columna + 1} baja: {TABLE_C[:, columna].tolist()}")
    assert not fallos, "tabla C no monótona, posible errata: " + "; ".join(fallos)


def test_table_a_is_monotonic_on_every_axis() -> None:
    """Tronco, cuello y piernas por separado. La tabla viene aplanada, se desdobla."""
    desdoblada = TABLE_A.reshape(5, 3, 4)  # tronco, cuello, piernas
    fallos = []
    for eje, nombre in ((0, "tronco"), (1, "cuello"), (2, "piernas")):
        diffs = np.diff(desdoblada, axis=eje)
        if (diffs < 0).any():
            malos = np.argwhere(diffs < 0)
            fallos.append(f"{nombre}: {len(malos)} descensos, primero en {malos[0].tolist()}")
    assert not fallos, "tabla A no monótona, posible errata: " + "; ".join(fallos)


def test_table_b_is_monotonic_on_every_axis() -> None:
    desdoblada = TABLE_B.reshape(6, 2, 3)  # brazo, antebrazo, muñeca
    fallos = []
    for eje, nombre in ((0, "brazo"), (1, "antebrazo"), (2, "muñeca")):
        diffs = np.diff(desdoblada, axis=eje)
        if (diffs < 0).any():
            malos = np.argwhere(diffs < 0)
            fallos.append(f"{nombre}: {len(malos)} descensos, primero en {malos[0].tolist()}")
    assert not fallos, "tabla B no monótona, posible errata: " + "; ".join(fallos)


# --- los umbrales de la norma en sus fronteras -------------------------------


def test_component_thresholds() -> None:
    assert trunk_score(np.array([0.0, 10, 20, 21, 60, 61, 90])).tolist() == [1, 2, 2, 3, 3, 4, 4]
    assert neck_score(np.array([0.0, 20, 21, 45])).tolist() == [1, 1, 2, 2]
    assert upper_arm_score(np.array([0.0, 20, 21, 45, 46, 90, 91])).tolist() == [
        1, 1, 2, 2, 3, 3, 4,
    ]
    assert lower_arm_score(np.array([0.0, 59, 60, 100, 101])).tolist() == [2, 2, 1, 1, 2]


def test_action_levels_cover_every_score() -> None:
    for puntaje in range(1, 16):
        assert action_level(puntaje)
    for fuera in (0, 16):
        try:
            action_level(fuera)
        except ValueError:
            continue
        raise AssertionError(f"aceptó un puntaje imposible: {fuera}")


# --- el cálculo completo ------------------------------------------------------


def _standing_neutral() -> np.ndarray:
    """Una persona de pie, erguida, brazos abajo, en coordenadas de imagen."""
    kp = np.zeros((1, N_JOINTS, 2), dtype=np.float32)
    kp[0, JOINT["left_hip"]] = (95, 300)
    kp[0, JOINT["right_hip"]] = (105, 300)
    kp[0, JOINT["left_shoulder"]] = (95, 200)
    kp[0, JOINT["right_shoulder"]] = (105, 200)
    kp[0, JOINT["left_ear"]] = (95, 160)
    kp[0, JOINT["right_ear"]] = (105, 160)
    kp[0, JOINT["right_elbow"]] = (105, 260)
    kp[0, JOINT["right_wrist"]] = (105, 320)
    kp[0, JOINT["right_knee"]] = (105, 400)
    kp[0, JOINT["right_ankle"]] = (105, 500)
    return kp


def _bent_over() -> np.ndarray:
    """Agachado a recoger algo del suelo: tronco casi horizontal, brazo adelantado."""
    kp = _standing_neutral().copy()
    kp[0, JOINT["left_shoulder"]] = (170, 290)
    kp[0, JOINT["right_shoulder"]] = (180, 290)
    kp[0, JOINT["left_ear"]] = (210, 300)
    kp[0, JOINT["right_ear"]] = (220, 300)
    kp[0, JOINT["right_elbow"]] = (190, 350)
    kp[0, JOINT["right_wrist"]] = (200, 410)
    kp[0, JOINT["right_knee"]] = (105, 380)
    kp[0, JOINT["right_ankle"]] = (110, 500)
    return kp


def test_neutral_posture_scores_every_component_as_the_norm_says() -> None:
    """Componente a componente, no solo el total.

    La primera versión de esta prueba solo miraba que el REBA agregado saliera
    bajo, y pasaba con la flexión de rodilla y de codo invertidas: una pierna recta
    daba 180° de flexión y puntuaba 3, pero el total se quedaba en 2 y la prueba lo
    dejaba pasar. Ver `docs/mediciones-falsas.md`, punto 3. El agregado tolera un
    componente roto; el desglose no.
    """
    r = reba_from_keypoints(_standing_neutral())
    angles = body_angles(_standing_neutral())
    assert abs(angles["knee_flexion"][0]) < 5, (
        f"una pierna recta no puede tener {angles['knee_flexion'][0]:.0f}° de flexión"
    )
    assert abs(angles["lower_arm_flexion"][0]) < 5, (
        f"un codo extendido no puede tener {angles['lower_arm_flexion'][0]:.0f}° de flexión"
    )
    assert int(r["trunk"][0]) == 1, f"tronco erguido: {r['trunk'][0]}"
    assert int(r["neck"][0]) == 1, f"cuello recto: {r['neck'][0]}"
    assert int(r["legs"][0]) == 1, f"piernas rectas con apoyo bilateral: {r['legs'][0]}"
    assert int(r["upper_arm"][0]) == 1, f"brazo pegado al cuerpo: {r['upper_arm'][0]}"
    # El antebrazo SÍ puntúa 2 aquí, y es correcto: la norma considera óptimo el
    # codo entre 60° y 100°, así que un brazo colgando extendido penaliza.
    assert int(r["lower_arm"][0]) == 2, f"codo extendido: {r['lower_arm'][0]}"
    assert action_level(int(r["reba"][0])) in ("despreciable", "bajo")


def test_bent_elbow_scores_better_than_extended_one() -> None:
    """El codo en escuadra es lo óptimo para la norma, y el extendido penaliza.

    Comprueba el ángulo del codo en el otro sentido: si estuviera invertido, esta
    prueba y la de arriba no podrían pasar las dos.
    """
    kp = _standing_neutral().copy()
    kp[0, JOINT["right_elbow"]] = (105, 260)
    kp[0, JOINT["right_wrist"]] = (165, 260)  # antebrazo horizontal: codo a 90°
    angles = body_angles(kp)
    assert 80 < angles["lower_arm_flexion"][0] < 100, (
        f"un codo en escuadra debería medir ~90°, midió {angles['lower_arm_flexion'][0]:.0f}°"
    )
    assert int(reba_from_keypoints(kp)["lower_arm"][0]) == 1


def test_bent_knee_is_detected() -> None:
    """Y la rodilla en el otro sentido, por el mismo motivo."""
    kp = _standing_neutral().copy()
    kp[0, JOINT["right_knee"]] = (105, 400)
    kp[0, JOINT["right_ankle"]] = (165, 400)  # pantorrilla horizontal: rodilla a 90°
    angles = body_angles(kp)
    assert 80 < angles["knee_flexion"][0] < 100, (
        f"una rodilla en escuadra debería medir ~90°, midió {angles['knee_flexion'][0]:.0f}°"
    )
    assert int(reba_from_keypoints(kp)["legs"][0]) == 3  # 1 bilateral + 2 por >60°


def test_bent_posture_scores_higher_than_neutral() -> None:
    neutro = reba_from_keypoints(_standing_neutral())["reba"][0]
    agachado = reba_from_keypoints(_bent_over())["reba"][0]
    assert agachado > neutro, (
        f"agacharse tiene que puntuar más que estar de pie: {agachado} contra {neutro}"
    )


def test_assumptions_can_only_raise_the_score() -> None:
    """Los valores por defecto son los neutros, así que el puntaje es una cota inferior.

    Si alguna asunción pudiera BAJAR el puntaje, la frase «cota inferior del riesgo
    real» que este proyecto va a publicar sería falsa.
    """
    kp = _bent_over()
    base = int(reba_from_keypoints(kp)["reba"][0])
    peores = [
        RebaAssumptions(wrist=3),
        RebaAssumptions(trunk_twist=True),
        RebaAssumptions(neck_twist=True),
        RebaAssumptions(load_force=2),
        RebaAssumptions(coupling=3),
        RebaAssumptions(wrist=3, trunk_twist=True, neck_twist=True, load_force=3, coupling=3),
    ]
    for supuesto in peores:
        puntaje = int(reba_from_keypoints(kp, assumptions=supuesto)["reba"][0])
        assert puntaje >= base, f"{supuesto} bajó el puntaje de {base} a {puntaje}"


def test_absent_frames_get_no_risk_level() -> None:
    """Sin persona no hay puntaje, y 0 no es un nivel de la norma: no se confunde
    con riesgo bajo."""
    kp = np.concatenate([_standing_neutral(), _bent_over()])
    present = np.array([True, False])
    r = reba_from_keypoints(kp, present=present)
    assert r["reba"][1] == 0
    try:
        action_level(int(r["reba"][1]))
    except ValueError:
        return
    raise AssertionError("el 0 de un fotograma sin persona pasó por un nivel válido")


def test_the_more_visible_side_is_the_one_measured() -> None:
    """Con un lado ocluido, se mide por el otro.

    Antes se usaba siempre el derecho, sin más motivo que haber elegido uno. Medido
    el 30/09 sobre una grabación con una caja delante del cuerpo, el codo y la
    muñeca del lado tapado caen por debajo de 0,5 en el 21-24% de los fotogramas.
    """
    from src.baseline.reba import best_side

    confianzas = np.full(N_JOINTS, 0.9, dtype=np.float32)
    for nombre in ("right_shoulder", "right_elbow", "right_wrist", "right_knee", "right_ankle"):
        confianzas[JOINT[nombre]] = 0.2
    assert best_side(confianzas) == "left"

    confianzas[:] = 0.9
    for nombre in ("left_elbow", "left_wrist", "left_ankle"):
        confianzas[JOINT[nombre]] = 0.15
    assert best_side(confianzas) == "right"


def test_a_component_whose_joints_are_hidden_is_flagged_not_invented() -> None:
    """Si una articulación no se ve, ese componente se marca como no fiable.

    Antes entraba en el cálculo igual que una perfectamente visible: un ángulo
    sacado de una muñeca con 0,1 de confianza pesaba lo mismo que uno real, y el
    puntaje salía con la misma cara de certeza.
    """
    kp = np.concatenate([_standing_neutral(), _standing_neutral()])
    confianzas = np.full((2, N_JOINTS), 0.9, dtype=np.float32)
    # En el segundo fotograma se tapan las dos muñecas: el antebrazo deja de ser
    # calculable por cualquiera de los dos lados.
    confianzas[1, JOINT["left_wrist"]] = 0.1
    confianzas[1, JOINT["right_wrist"]] = 0.1

    r = reba_from_keypoints(kp, scores=confianzas)
    assert bool(r["reliable"]["lower_arm"][0]) is True
    assert bool(r["reliable"]["lower_arm"][1]) is False
    assert bool(r["partial"][0]) is False and bool(r["partial"][1]) is True
    # El tronco sigue siendo fiable: sus articulaciones se ven en los dos.
    assert bool(r["reliable"]["trunk"][1]) is True


def test_without_confidences_everything_behaves_as_before() -> None:
    """La compatibilidad hacia atrás: sin confianzas, el resultado no cambia.

    Importa porque todos los números publicados del proyecto se calcularon así, y un
    cambio silencioso en el cálculo los invalidaría sin avisar.
    """
    kp = _bent_over()
    antes = reba_from_keypoints(kp)
    assert int(antes["reba"][0]) > 0
    assert bool(antes["partial"][0]) is False, "sin confianzas nada se marca como parcial"
    assert antes["side"][0] == "right", "sin confianzas se conserva el lado de siempre"


def test_invalid_assumptions_are_rejected() -> None:
    for kwargs in ({"wrist": 0}, {"wrist": 4}, {"load_force": -1}, {"coupling": 9}):
        try:
            RebaAssumptions(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"aceptó asunciones imposibles: {kwargs}")


def test_score_combination_matches_tables_directly() -> None:
    """Que los índices de las tablas no estén cruzados, comprobado celda a celda."""
    for trunk in range(1, 6):
        for neck in range(1, 4):
            for legs in range(1, 5):
                esperado = TABLE_A.reshape(5, 3, 4)[trunk - 1, neck - 1, legs - 1]
                obtenido = score_a(np.array([trunk]), np.array([neck]), np.array([legs]))[0]
                assert obtenido == esperado, f"A({trunk},{neck},{legs}): {obtenido} != {esperado}"
    for arm in range(1, 7):
        for forearm in range(1, 3):
            for wrist in range(1, 4):
                esperado = TABLE_B.reshape(6, 2, 3)[arm - 1, forearm - 1, wrist - 1]
                obtenido = score_b(np.array([arm]), np.array([forearm]), np.array([wrist]))[0]
                assert obtenido == esperado, f"B({arm},{forearm},{wrist}): {obtenido} != {esperado}"


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
