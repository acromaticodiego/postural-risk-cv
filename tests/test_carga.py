"""La fusión de persona y carga, comprobada sin GPU y sin modelo.

Toda la lógica que decide algo —si una caja está en las manos, cuánto mide, qué peso
se le aplica— es geometría pura y se prueba aquí. La inferencia queda fuera a
propósito: es la parte que no toma decisiones.

Las dos que más protegen:

  · `test_a_box_lying_around_is_not_the_load`: asociar a la fuerza la caja más
    cercana haría que el sistema aplicara un peso a alguien que solo camina, y ese
    peso entra en el cálculo de riesgo.
  · `test_a_size_outside_the_catalog_matches_nothing`: si el cliente declaró cajas
    de 30 y 45 cm y la cámara mide 80, lo honesto es decir que eso no está en el
    catálogo, no asignarle el peso de la más grande y seguir.

    .\\.venv\\Scripts\\python.exe -m tests.test_carga
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.load.carga import (  # noqa: E402
    LoadCatalogEntry,
    associate_to_person,
    body_height_px,
    build_load,
    hands_center,
    load_size_cm,
    match_catalog,
)
from src.pose.schema import JOINT, N_JOINTS  # noqa: E402

CATALOGO = (
    LoadCatalogEntry("caja pequena", width_cm=28.0, weight_kg=5.0),
    LoadCatalogEntry("caja grande", width_cm=48.0, weight_kg=18.0),
)


def _persona(manos=(400.0, 500.0)) -> np.ndarray:
    """Una persona de 500 px de alto con las manos donde se diga."""
    kp = np.zeros((N_JOINTS, 2), dtype=np.float32)
    kp[JOINT["nose"]] = (400, 150)
    kp[JOINT["left_shoulder"]] = (380, 220)
    kp[JOINT["right_shoulder"]] = (420, 220)
    kp[JOINT["left_hip"]] = (385, 380)
    kp[JOINT["right_hip"]] = (415, 380)
    kp[JOINT["left_wrist"]] = (manos[0] - 20, manos[1])
    kp[JOINT["right_wrist"]] = (manos[0] + 20, manos[1])
    kp[JOINT["left_ankle"]] = (385, 650)
    kp[JOINT["right_ankle"]] = (415, 650)
    return kp


def _caja(centro, ancho=100.0, alto=80.0) -> list[float]:
    return [centro[0] - ancho / 2, centro[1] - alto / 2, centro[0] + ancho / 2, centro[1] + alto / 2]


# --- asociación: qué caja es LA carga ----------------------------------------


def test_the_box_in_the_hands_is_the_one_chosen() -> None:
    persona = _persona(manos=(400.0, 500.0))
    cajas = np.array([_caja((400, 505)), _caja((900, 300))])
    assert associate_to_person(cajas, persona) == 0


def test_a_box_lying_around_is_not_the_load() -> None:
    """Si no hay ninguna caja en las manos, la respuesta es NINGUNA.

    Asignarle a la fuerza la más próxima haría que el sistema aplicara un peso a
    alguien que solo está caminando, y ese peso entra en el cálculo de riesgo.
    """
    persona = _persona(manos=(400.0, 500.0))
    lejos = np.array([_caja((1200, 900)), _caja((50, 80))])
    assert associate_to_person(lejos, persona) is None


def test_with_several_boxes_near_the_closest_wins() -> None:
    persona = _persona(manos=(400.0, 500.0))
    cajas = np.array([_caja((460, 520)), _caja((402, 502)), _caja((520, 540))])
    assert associate_to_person(cajas, persona) == 1


def test_no_boxes_at_all_is_handled() -> None:
    persona = _persona()
    assert associate_to_person(np.empty((0, 4)), persona) is None
    assert associate_to_person(None, persona) is None


def test_association_does_not_depend_on_distance_to_the_camera() -> None:
    """La misma escena, la persona el doble de grande, tiene que decidir igual.

    El radio va en alturas de cuerpo justamente para eso: si fuera en píxeles, una
    cámara más cerca o más lejos cambiaría qué se considera «en las manos».
    """
    pequena = _persona(manos=(400.0, 500.0))
    caja_pequena = np.array([_caja((400, 530), ancho=100, alto=80)])
    assert associate_to_person(caja_pequena, pequena) == 0

    grande = pequena * 2.0
    caja_grande = np.array([_caja((800, 1060), ancho=200, alto=160)])
    assert associate_to_person(caja_grande, grande) == 0, (
        "la misma escena al doble de tamaño tiene que dar la misma respuesta"
    )


# --- medida y catálogo -------------------------------------------------------


def test_size_in_centimetres_uses_the_body_as_a_ruler() -> None:
    # 500 px de cuerpo para 170 cm -> unos 2,94 px/cm. Una caja de 100 px son ~34 cm.
    px_cm = 500 / 170
    ancho, alto = load_size_cm(_caja((400, 500), ancho=100, alto=80), px_cm)
    assert 32 < ancho < 36, ancho
    assert 25 < alto < 29, alto


def test_catalog_matches_by_width() -> None:
    assert match_catalog(30.0, CATALOGO).name == "caja pequena"
    assert match_catalog(46.0, CATALOGO).name == "caja grande"


def test_a_size_outside_the_catalog_matches_nothing() -> None:
    """Fuera de la tolerancia, None. No la más parecida."""
    assert match_catalog(90.0, CATALOGO) is None
    assert match_catalog(5.0, CATALOGO) is None


def test_without_catalog_there_is_no_weight() -> None:
    carga = build_load(_caja((400, 500)), 0.9, px_per_cm=500 / 170, catalog=None)
    assert carga.matched is None and carga.weight_kg is None
    assert carga.width_cm is not None, "el tamaño sí se mide aunque no haya catálogo"


def test_without_scale_there_is_no_size() -> None:
    """Sin estatura declarada no hay escala, y sin escala no se inventan centímetros."""
    carga = build_load(_caja((400, 500)), 0.9, px_per_cm=None, catalog=CATALOGO)
    assert carga.width_cm is None and carga.matched is None
    assert carga.center == (400.0, 500.0), "el centro en píxeles sí se conoce siempre"


def test_build_load_carries_the_weight_of_the_matched_entry() -> None:
    carga = build_load(
        _caja((400, 500), ancho=140), 0.88, px_per_cm=500 / 170, catalog=CATALOGO
    )
    assert carga.matched is not None
    assert carga.weight_kg == carga.matched.weight_kg


# --- utilidades --------------------------------------------------------------


def test_hands_and_height() -> None:
    persona = _persona(manos=(400.0, 500.0))
    assert np.allclose(hands_center(persona), (400.0, 500.0))
    assert 490 < body_height_px(persona) < 510


def test_missing_joints_do_not_crash() -> None:
    vacia = np.zeros((N_JOINTS, 2), dtype=np.float32)
    assert body_height_px(vacia) == 1.0
    assert associate_to_person(np.array([_caja((10, 10))]), vacia) is None


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
