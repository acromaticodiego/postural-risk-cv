"""La ecuación NIOSH, y la prueba que justifica que exista.

`test_technique_changes_niosh_much_more_than_reba` es la razón de ser del módulo:
con REBA, levantar con la espalda doblada y levantar con las rodillas dobladas se
diferenciaban en UN punto (7 contra 6), porque la norma sube el puntaje de piernas
casi tanto como baja el de tronco. Si NIOSH tampoco distinguiera las dos técnicas,
añadirlo no serviría de nada y habría que decirlo en vez de quedárselo.

    .\\.venv\\Scripts\\python.exe -m tests.test_niosh
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline.niosh import (  # noqa: E402
    LOAD_CONSTANT_KG,
    _pixels_per_cm,
    analyze_lift,
    asymmetry_multiplier,
    distance_multiplier,
    frequency_multiplier,
    horizontal_multiplier,
    pixels_per_cm_from_sequence,
    vertical_multiplier,
)
from src.baseline.reba import reba_from_keypoints  # noqa: E402
from src.pose.schema import JOINT, N_JOINTS  # noqa: E402
from src.product.workstation import WorkstationConfig  # noqa: E402

HZ = 10.0


def _cuerpo(**puntos) -> np.ndarray:
    kp = np.zeros((1, N_JOINTS, 2), dtype=np.float32)
    for nombre, valor in puntos.items():
        kp[0, JOINT[nombre]] = valor
    return kp


def _secuencia(origen: np.ndarray, destino: np.ndarray, n: int = 12) -> np.ndarray:
    """Un levantamiento: del esqueleto de origen al de destino."""
    pasos = np.linspace(0.0, 1.0, n)[:, None, None]
    return (origen[0][None, ...] * (1 - pasos) + destino[0][None, ...] * pasos).astype(np.float32)


# Dos formas de levantar la MISMA caja del suelo hasta la cintura. Lo único que
# cambia es dónde están las manos respecto a los pies: pegadas al cuerpo o lejos.
# Las escalas son píxeles con una persona de 500 px de alto ~ 170 cm.
_PIES_X = 300.0


def _origen(distancia_horizontal_px: float) -> np.ndarray:
    manos_x = _PIES_X + distancia_horizontal_px
    return _cuerpo(
        nose=(_PIES_X + distancia_horizontal_px * 0.5, 180),
        left_shoulder=(_PIES_X + distancia_horizontal_px * 0.45, 230),
        right_shoulder=(_PIES_X + distancia_horizontal_px * 0.45 + 10, 230),
        left_hip=(_PIES_X, 380), right_hip=(_PIES_X + 10, 380),
        left_wrist=(manos_x, 600), right_wrist=(manos_x + 10, 600),
        left_elbow=(manos_x - 10, 500), right_elbow=(manos_x, 500),
        left_knee=(_PIES_X - 5, 500), right_knee=(_PIES_X + 15, 500),
        left_ankle=(_PIES_X - 5, 640), right_ankle=(_PIES_X + 15, 640),
    )


def _destino() -> np.ndarray:
    return _cuerpo(
        nose=(_PIES_X + 5, 150),
        left_shoulder=(_PIES_X, 200), right_shoulder=(_PIES_X + 10, 200),
        left_hip=(_PIES_X, 340), right_hip=(_PIES_X + 10, 340),
        left_wrist=(_PIES_X + 15, 350), right_wrist=(_PIES_X + 25, 350),
        left_elbow=(_PIES_X + 5, 280), right_elbow=(_PIES_X + 15, 280),
        left_knee=(_PIES_X - 5, 480), right_knee=(_PIES_X + 15, 480),
        left_ankle=(_PIES_X - 5, 640), right_ankle=(_PIES_X + 15, 640),
    )


# --- la escala, que es de donde salen todos los centímetros -------------------


def test_bending_shrinks_the_body_in_the_image_but_not_the_person() -> None:
    """La misma persona, a la misma distancia, da escalas distintas según se agache.

    Es el fallo que obliga a tomar la escala de la secuencia y no del fotograma: medir
    de la coronilla al tobillo mide la PROYECCIÓN del cuerpo, y esa se encoge al
    doblarse. Medido en una grabación real del 2026-10-02, 2,16 a 4,41 px/cm en el
    mismo vídeo.
    """
    agachado = _origen(distancia_horizontal_px=160.0)[0]
    erguido = _destino()[0]
    assert _pixels_per_cm(agachado, 170.0) < _pixels_per_cm(erguido, 170.0)


def test_the_scale_of_a_lift_comes_from_its_most_upright_frame() -> None:
    """Y no del fotograma de origen, que es el más agachado del evento — justo el peor
    sitio para preguntarle la escala al cuerpo, y justo donde NIOSH mide todo lo demás."""
    secuencia = _secuencia(_origen(distancia_horizontal_px=160.0), _destino())
    de_la_secuencia = pixels_per_cm_from_sequence(secuencia, 170.0)
    del_origen = _pixels_per_cm(secuencia[0], 170.0)
    assert de_la_secuencia > del_origen
    assert de_la_secuencia == max(_pixels_per_cm(k, 170.0) for k in secuencia)


def test_a_single_frame_still_works() -> None:
    uno = _destino()[0]
    assert pixels_per_cm_from_sequence(uno, 170.0) == _pixels_per_cm(uno, 170.0)


def test_the_lift_analysis_actually_uses_the_sequence_scale() -> None:
    """Y esta es la que protege de verdad.

    Las dos de arriba comprueban la FUNCIÓN de la escala; revertir `analyze_lift` para
    que volviera a tomarla del fotograma de origen las dejaba a las dos en verde. Una
    prueba que sigue pasando cuando rompes lo que dice proteger no está protegiendo
    nada, y en este proyecto ya van cuatro veces.

    Aquí se comprueba el resultado: las distancias que publica `analyze_lift` tienen
    que salir de la escala de la secuencia, no de la del origen. Con la escala del
    origen —más pequeña, porque la persona está doblada— los centímetros salen
    inflados por el mismo factor.
    """
    secuencia = _secuencia(_origen(distancia_horizontal_px=160.0), _destino())
    analisis = analyze_lift(secuencia, HZ, worker_height_cm=170.0, load_kg=10.0)

    escala_secuencia = pixels_per_cm_from_sequence(secuencia, 170.0)
    escala_origen = _pixels_per_cm(secuencia[0], 170.0)
    assert escala_origen < escala_secuencia, "sin esto la prueba no distingue nada"

    manos = secuencia[:, [JOINT["left_wrist"], JOINT["right_wrist"]], :].mean(axis=1)
    i = int(np.argmax(manos[:, 1]))
    pies_x = float(
        np.mean(secuencia[i][[JOINT["left_ankle"], JOINT["right_ankle"]], 0])
    )
    distancia_px = abs(float(manos[i, 0]) - pies_x)

    bien = distancia_px / escala_secuencia
    mal = distancia_px / escala_origen
    assert abs(analisis.horizontal_cm - bien) < 0.5
    assert abs(analisis.horizontal_cm - mal) > 1.0


# --- la prueba que justifica el módulo ---------------------------------------


def test_technique_changes_niosh_much_more_than_reba() -> None:
    """Acercar la carga al cuerpo tiene que notarse mucho, porque es lo que importa."""
    cerca = analyze_lift(_secuencia(_origen(25), _destino()), HZ, worker_height_cm=170, load_kg=12)
    lejos = analyze_lift(_secuencia(_origen(140), _destino()), HZ, worker_height_cm=170, load_kg=12)

    assert lejos.horizontal_cm > cerca.horizontal_cm + 20, (
        f"los dos casos no se diferencian en distancia: {cerca.horizontal_cm} y {lejos.horizontal_cm}"
    )
    assert lejos.lifting_index > cerca.lifting_index * 1.5, (
        "NIOSH tampoco distingue las dos técnicas, así que añadirlo no aporta: "
        f"índice {cerca.lifting_index} cerca contra {lejos.lifting_index} lejos"
    )
    assert lejos.worst_factor == "horizontal", (
        f"con la carga lejos, el factor que más penaliza debería ser el horizontal, "
        f"no {lejos.worst_factor}"
    )


def test_what_niosh_adds_over_reba_is_an_answer_in_kilos() -> None:
    """Lo que NIOSH aporta, comprobado en vez de supuesto.

    La primera versión de esta prueba afirmaba que REBA NO separa las dos técnicas y
    NIOSH sí. Medido, era falso: al alejar la carga también se inclina el tronco y
    se eleva el brazo, así que REBA sube igual (de 3 a 6). Generalicé a partir de un
    caso concreto —espalda doblada contra rodillas dobladas con las manos en el
    mismo sitio— donde REBA se movía un punto, y no era el caso general.

    Lo que NIOSH sí aporta, y esto es lo que se comprueba aquí:

      · un resultado **en kilogramos**: cuánto debería pesar la caja para que ese
        levantamiento fuera aceptable. Un puntaje ordinal de 6 no le dice a nadie
        qué hacer; «esta caja de 12 kg debería pesar 8» sí.
      · un **umbral de aceptabilidad** con el peso real dentro: el índice cruza 1
        cuando el levantamiento deja de ser recomendable, y REBA no tiene dónde
        meter el peso salvo un ajuste de 0 a 3.
    """
    config = WorkstationConfig("a", "a", load_kg=12, coupling="fair")
    cerca_reba = int(reba_from_keypoints(_origen(25), assumptions=config.assumptions())["reba"][0])
    lejos_reba = int(reba_from_keypoints(_origen(140), assumptions=config.assumptions())["reba"][0])
    assert lejos_reba > cerca_reba, "REBA también detecta alejar la carga, y eso está bien"

    cerca = analyze_lift(_secuencia(_origen(25), _destino()), HZ, worker_height_cm=170, load_kg=12)
    lejos = analyze_lift(_secuencia(_origen(140), _destino()), HZ, worker_height_cm=170, load_kg=12)

    # En kilos: el peso aceptable cae al alejar la carga, y eso se puede llevar a
    # una reunión. REBA da un número ordinal que no se traduce a ninguna acción.
    assert lejos.recommended_weight_kg < cerca.recommended_weight_kg * 0.75, (
        f"el peso recomendado apenas baja: {cerca.recommended_weight_kg} -> "
        f"{lejos.recommended_weight_kg} kg"
    )
    # Y el umbral: cerca es aceptable y lejos no.
    assert cerca.lifting_index <= 1.0 < lejos.lifting_index, (
        f"el indice no cruza el umbral: {cerca.lifting_index} y {lejos.lifting_index}"
    )
    assert "dentro" in cerca.verdict and "dentro" not in lejos.verdict


# --- los multiplicadores, en las fronteras de la norma -----------------------


def test_horizontal_multiplier() -> None:
    assert horizontal_multiplier(20) == 1.0
    assert horizontal_multiplier(25) == 1.0
    assert abs(horizontal_multiplier(50) - 0.5) < 1e-9
    assert horizontal_multiplier(64) == 0.0, "por encima de 63 cm la norma lo da por inaceptable"


def test_vertical_multiplier_is_best_at_knuckle_height() -> None:
    assert abs(vertical_multiplier(75) - 1.0) < 1e-9
    assert vertical_multiplier(0) < vertical_multiplier(75)
    assert vertical_multiplier(150) < vertical_multiplier(75)
    assert vertical_multiplier(180) == 0.0


def test_distance_and_asymmetry_and_frequency() -> None:
    assert distance_multiplier(10) == 1.0
    assert distance_multiplier(180) == 0.0
    assert abs(asymmetry_multiplier(0) - 1.0) < 1e-9
    assert asymmetry_multiplier(140) == 0.0
    assert frequency_multiplier(0.2) == 1.0
    assert frequency_multiplier(20) == 0.0
    assert frequency_multiplier(8) < frequency_multiplier(1)


def test_perfect_lift_allows_the_load_constant() -> None:
    """Con todos los factores óptimos, el peso recomendado es la constante de 23 kg."""
    analisis = analyze_lift(
        _secuencia(_origen(25), _destino()),
        HZ,
        worker_height_cm=170,
        load_kg=None,
        coupling="good",
        asymmetry_deg=0.0,
        lifts_per_min=0.2,
    )
    assert analisis.recommended_weight_kg <= LOAD_CONSTANT_KG + 1e-6
    assert analisis.lifting_index is None, "sin peso declarado no puede haber índice"
    assert "sin peso declarado" in analisis.verdict


def test_measuring_at_the_load_centre_is_marked_and_changes_the_distance() -> None:
    """Con la carga detectada, la distancia se mide donde la norma pide.

    Sin ella se usa el punto medio de las muñecas, que SUBESTIMA: el centro de una
    caja queda por delante de los nudillos. Que el resultado diga por cuál de las
    dos vías se midió es parte de poder confiar en el número.
    """
    secuencia = _secuencia(_origen(60), _destino())
    manos = secuencia[:, [JOINT["left_wrist"], JOINT["right_wrist"]], :].mean(axis=1)
    # El centro de la caja, 25 px más allá de las manos y alejándose del cuerpo.
    centros = manos + np.array([25.0, 0.0])

    sin_carga = analyze_lift(secuencia, HZ, worker_height_cm=170, load_kg=12)
    con_carga = analyze_lift(secuencia, HZ, worker_height_cm=170, load_kg=12, load_centers=centros)

    assert not sin_carga.measured_at_load and con_carga.measured_at_load
    assert con_carga.horizontal_cm > sin_carga.horizontal_cm, (
        "medir al centro de la carga tiene que dar una distancia mayor que a las muñecas: "
        f"{sin_carga.horizontal_cm} y {con_carga.horizontal_cm}"
    )
    assert con_carga.lifting_index >= sin_carga.lifting_index, (
        "y por tanto un índice igual o peor, nunca mejor"
    )


def test_load_centres_of_the_wrong_length_are_ignored() -> None:
    """Una lista de centros que no cuadra con la secuencia se descarta en vez de
    emparejarse a medias: mezclar fotogramas daría una distancia de otro instante."""
    secuencia = _secuencia(_origen(60), _destino())
    analisis = analyze_lift(
        secuencia, HZ, worker_height_cm=170, load_kg=12, load_centers=np.zeros((3, 2))
    )
    assert not analisis.measured_at_load


def test_unknown_worker_height_is_flagged() -> None:
    """Un número en centímetros sacado de una estatura inventada tiene que decirlo."""
    con = analyze_lift(_secuencia(_origen(60), _destino()), HZ, worker_height_cm=170, load_kg=12)
    sin = analyze_lift(_secuencia(_origen(60), _destino()), HZ, load_kg=12)
    assert not con.scale_estimated and sin.scale_estimated


def test_verdict_matches_the_index() -> None:
    cerca = analyze_lift(_secuencia(_origen(25), _destino()), HZ, worker_height_cm=170, load_kg=5)
    lejos = analyze_lift(_secuencia(_origen(150), _destino()), HZ, worker_height_cm=170, load_kg=25)
    assert cerca.lifting_index <= 1.0 and "dentro" in cerca.verdict
    assert lejos.lifting_index > 1.0 and "recomendado" in lejos.verdict or "riesgo" in lejos.verdict


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
