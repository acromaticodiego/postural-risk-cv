"""El pre-etiquetado de la carga, comprobado sin GPU y sin SAM.

SAM queda fuera a propósito: no toma ninguna decisión. Las decisiones son dónde se
ponen los puntos de aviso, qué máscara se acepta y cómo se convierte en etiqueta, y
todo eso es geometría.

Las tres que más protegen, y cada una tapa un fallo que ya se ha visto hacer a SAM
guiado por puntos:

  · `test_a_mask_that_swallows_the_person_is_rejected`: la máscara que se lleva el
    torso es la más peligrosa de todas, porque de lejos parece una caja grande y
    pasa la validación humana de un vistazo. Si entra, el modelo afinado aprende a
    segmentar personas y lo hará con confianza alta.
  · `test_scattered_blobs_are_rejected`: trozos sueltos que suman un área razonable.
    Un filtro que solo mirara el área los aceptaría.
  · `test_varied_frames_prefer_different_postures_over_different_instants`: si la
    selección de fotogramas se queda con instantes repartidos en vez de posturas
    distintas, el conjunto de afinado es la misma imagen veinte veces.

    .\\.venv\\Scripts\\python.exe -m tests.test_prelabel
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.load.prelabel import (  # noqa: E402
    build_prompt,
    judge_mask,
    largest_component_fraction,
    mask_to_polygon,
    pick_varied,
    polygon_to_yolo_label,
    pose_vector,
)
from src.pose.schema import JOINT, N_JOINTS  # noqa: E402

ALTO, ANCHO = 480, 640


def _persona(manos=(320.0, 260.0)) -> tuple[np.ndarray, np.ndarray]:
    """Una persona de pie de unos 300 px, con las manos donde se diga, toda visible."""
    kp = np.zeros((N_JOINTS, 2), dtype=np.float32)
    kp[JOINT["nose"]] = (320, 90)
    kp[JOINT["left_shoulder"]] = (295, 140)
    kp[JOINT["right_shoulder"]] = (345, 140)
    kp[JOINT["left_hip"]] = (300, 240)
    kp[JOINT["right_hip"]] = (340, 240)
    kp[JOINT["left_knee"]] = (300, 310)
    kp[JOINT["right_knee"]] = (340, 310)
    kp[JOINT["left_ankle"]] = (300, 380)
    kp[JOINT["right_ankle"]] = (340, 380)
    kp[JOINT["left_wrist"]] = (manos[0] - 30, manos[1])
    kp[JOINT["right_wrist"]] = (manos[0] + 30, manos[1])
    scores = np.full(N_JOINTS, 0.9, dtype=np.float32)
    return kp, scores


def _mascara(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros((ALTO, ANCHO), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


# --- los puntos de aviso -----------------------------------------------------


def test_the_positive_point_goes_between_the_hands() -> None:
    kp, scores = _persona(manos=(320.0, 260.0))
    aviso = build_prompt(kp, scores)
    assert aviso is not None
    assert aviso.points[0] == [320.0, 260.0]
    assert aviso.labels[0] == 1


def test_the_body_is_marked_as_not_the_load() -> None:
    kp, scores = _persona()
    aviso = build_prompt(kp, scores)
    assert aviso is not None
    assert aviso.positives == 1
    # Nariz, dos hombros, dos caderas y dos rodillas.
    assert aviso.negatives == 7


def test_without_visible_wrists_there_is_no_prompt() -> None:
    """Sin saber dónde están las manos, el punto positivo se pondría en un sitio
    inventado y la máscara saldría de cualquier cosa. Mejor descartar el fotograma."""
    kp, scores = _persona()
    scores[JOINT["left_wrist"]] = 0.2
    assert build_prompt(kp, scores) is None


def test_low_confidence_joints_do_not_become_negative_points() -> None:
    """Un negativo mal puesto es peor que no ponerlo: si cae sobre la caja, le dice a
    SAM que la caja no es la caja."""
    kp, scores = _persona()
    for nombre in ("nose", "left_hip", "right_hip", "left_knee", "right_knee"):
        scores[JOINT[nombre]] = 0.1
    aviso = build_prompt(kp, scores)
    assert aviso is not None
    assert aviso.negatives == 2  # solo los dos hombros


def test_a_body_with_no_visible_joints_gives_no_prompt() -> None:
    """Sin un solo negativo SAM se lleva a la persona entera, porque al levantar la
    caja va pegada al cuerpo. Antes no proponer nada que proponer un contorno."""
    kp, scores = _persona()
    scores[:] = 0.1
    scores[JOINT["left_wrist"]] = scores[JOINT["right_wrist"]] = 0.9
    assert build_prompt(kp, scores) is None


# --- qué máscara se acepta ---------------------------------------------------


def test_a_box_in_the_hands_is_accepted() -> None:
    kp, scores = _persona(manos=(320.0, 260.0))
    veredicto = judge_mask(_mascara(270, 230, 370, 300), kp, scores)
    assert veredicto.ok, veredicto.reason


def test_a_box_against_the_belly_is_accepted_even_though_it_hides_the_hips() -> None:
    """La primera versión de este filtro rechazaba el caso bueno: una caja sujetada
    contra el vientre TAPA las caderas en la imagen, y eso es lo normal al levantar."""
    kp, scores = _persona(manos=(320.0, 250.0))
    veredicto = judge_mask(_mascara(280, 225, 365, 285), kp, scores)
    assert veredicto.ok, veredicto.reason


def test_a_mask_that_swallows_the_person_is_rejected() -> None:
    """La silueta entera: tan alta como la persona y tapando la cara."""
    kp, scores = _persona(manos=(320.0, 260.0))
    veredicto = judge_mask(_mascara(280, 80, 360, 390), kp, scores)
    assert not veredicto.ok
    assert "cara" in veredicto.reason


def test_a_mask_as_tall_as_the_person_is_rejected_even_below_the_face() -> None:
    """Y sin llegar a la cara tampoco pasa: una caja de 50 cm en las manos de alguien
    de 1,70 m no ocupa el alto de su cuerpo."""
    kp, scores = _persona(manos=(320.0, 260.0))
    veredicto = judge_mask(_mascara(285, 140, 355, 385), kp, scores)
    assert not veredicto.ok
    assert "persona" in veredicto.reason


def test_a_mask_far_from_the_hands_is_rejected() -> None:
    """Una caja en el suelo al otro lado de la escena no es la carga de nadie."""
    kp, scores = _persona(manos=(320.0, 260.0))
    veredicto = judge_mask(_mascara(20, 400, 120, 460), kp, scores)
    assert not veredicto.ok
    assert "manos" in veredicto.reason


def test_a_tiny_mask_is_rejected() -> None:
    kp, scores = _persona()
    veredicto = judge_mask(_mascara(315, 255, 325, 265), kp, scores)
    assert not veredicto.ok
    assert "pequena" in veredicto.reason


def test_half_the_scene_is_rejected() -> None:
    kp, scores = _persona()
    veredicto = judge_mask(_mascara(0, 0, ANCHO, ALTO // 2), kp, scores)
    assert not veredicto.ok
    assert "grande" in veredicto.reason


def test_scattered_blobs_are_rejected() -> None:
    """Cuatro manchas que suman un área plausible. Un filtro de área sola las deja
    pasar, y no son ninguna caja: son SAM siguiendo una textura."""
    kp, scores = _persona(manos=(320.0, 260.0))
    m = np.zeros((ALTO, ANCHO), dtype=np.uint8)
    for x in (260, 300, 340, 380):
        m[240:270, x : x + 25] = 1
    veredicto = judge_mask(m, kp, scores)
    assert not veredicto.ok
    assert "trozos" in veredicto.reason


def test_one_piece_with_a_speck_still_counts_as_one_piece() -> None:
    """Una mota suelta de ruido no convierte una caja buena en trozos sueltos."""
    m = _mascara(270, 230, 370, 300)
    m[400, 600] = 1
    assert largest_component_fraction(m) > 0.99


# --- la etiqueta que se le entrega a YOLO ------------------------------------


def test_a_rectangle_becomes_a_four_corner_polygon() -> None:
    poligono = mask_to_polygon(_mascara(100, 100, 300, 200))
    assert poligono is not None
    assert len(poligono) == 4


def test_the_label_is_normalized_and_inside_the_frame() -> None:
    poligono = mask_to_polygon(_mascara(100, 100, 300, 200))
    linea = polygon_to_yolo_label(poligono, ANCHO, ALTO)
    campos = linea.split()
    assert campos[0] == "0"
    valores = [float(v) for v in campos[1:]]
    assert len(valores) == 8
    assert all(0.0 <= v <= 1.0 for v in valores)
    # El contorno pasa por los CENTROS de los píxeles, así que el borde derecho de un
    # rectángulo que llega hasta la columna 300 sin incluirla es la columna 299.
    # La tolerancia es la del formato: la etiqueta se escribe con 6 decimales, que a
    # 640 px de ancho son 6 diezmilésimas de píxel.
    assert abs(max(valores[0::2]) - 299 / ANCHO) < 1e-5
    assert abs(max(valores[1::2]) - 199 / ALTO) < 1e-5


def test_an_empty_mask_has_no_polygon() -> None:
    assert mask_to_polygon(np.zeros((ALTO, ANCHO), dtype=np.uint8)) is None


# --- elegir fotogramas variados ----------------------------------------------


def test_varied_frames_prefer_different_postures_over_different_instants() -> None:
    """Veinte fotogramas de la misma postura y dos distintos al final. Elegir tres
    por reparto temporal devolvería tres veces lo mismo; por variedad, tienen que
    salir los dos distintos."""
    quieto, _ = _persona(manos=(320.0, 260.0))
    agachado, _ = _persona(manos=(320.0, 400.0))
    brazo_alto, _ = _persona(manos=(320.0, 120.0))
    caja = np.array([200.0, 60.0, 440.0, 400.0])

    vectores = [pose_vector(quieto, caja) for _ in range(20)]
    vectores += [pose_vector(agachado, caja), pose_vector(brazo_alto, caja)]
    tiempos = [float(i) for i in range(22)]

    elegidos = pick_varied(vectores, tiempos, n=3, min_gap_s=0.5)
    assert len(elegidos) == 3
    assert 20 in elegidos and 21 in elegidos


def test_the_minimum_gap_is_respected() -> None:
    """Dos fotogramas separados por 40 ms pueden medir distinto por el ruido del
    detector sin ser posturas distintas."""
    kp, _ = _persona()
    caja = np.array([200.0, 60.0, 440.0, 400.0])
    vectores = [pose_vector(kp + i * 3.0, caja) for i in range(10)]
    tiempos = [i * 0.04 for i in range(10)]

    elegidos = pick_varied(vectores, tiempos, n=5, min_gap_s=1.0)
    assert len(elegidos) == 1  # ninguno cumple la separación con el primero


def test_pose_vector_does_not_change_when_the_person_walks_closer() -> None:
    """Acercarse a la cámara no es cambiar de postura. Si lo fuera, la selección por
    variedad se llenaría de la misma pose a tres distancias."""
    kp, _ = _persona()
    caja = np.array([200.0, 60.0, 440.0, 400.0])
    lejos = pose_vector(kp, caja)
    cerca = pose_vector(kp * 2.0, caja * 2.0)
    assert np.allclose(lejos, cerca, atol=1e-5)


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
