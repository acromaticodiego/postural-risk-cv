"""El modo en vivo, comprobado sin necesitar a nadie delante de la cámara.

Lo que no se puede probar aquí es que la webcam dé imagen y que YOLO encuentre a la
persona: eso se comprobó a mano (la cámara abre y el bucle corre a 74 ms por
fotograma, 13 fps, con los 10 Hz que pide el modelo de sobra). Todo lo demás —el
puntaje, la predicción de la tarea, el cierre de eventos y el aviso de que el
modelo necesita contexto— sí, alimentando la sesión con esqueletos reales del
dataset.

    .\\.venv\\Scripts\\python.exe -m tests.test_live
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.harness import load  # noqa: E402
from src.live.session import TARGET_HZ, LiveSession  # noqa: E402
from src.product.workstation import WorkstationConfig  # noqa: E402

MODELO = Path(__file__).resolve().parents[1] / "artifacts/modelo"


def _sesion(**kwargs):
    config = WorkstationConfig(id="PRUEBA", name="Prueba", **kwargs)
    return LiveSession(config, camera=None)


def _caja(keypoints: np.ndarray) -> np.ndarray:
    xs, ys = keypoints[:, 0], keypoints[:, 1]
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()])


def test_model_is_trained_and_matches_the_session_rate() -> None:
    """El modelo se entrenó a 10 Hz y la sesión procesa a 10 Hz.

    Si alguien subiera la tasa de proceso para que el vídeo se viera más fluido, el
    modelo recibiría el gesto acelerado y empeoraría sin dar ningún error. Por eso
    la sesión se niega a arrancar si las dos no coinciden.
    """
    assert (MODELO / "tcn.pt").exists(), "falta el modelo: scripts/train_production_model.py"
    s = _sesion()
    assert abs(s.hz - TARGET_HZ) < 1e-6
    assert s.window_frames == int(round(2.0 * TARGET_HZ))


def test_no_task_is_invented_before_there_is_context() -> None:
    """Con menos de dos segundos de historia, el sistema dice que espera.

    Contestar una tarea con media ventana daría una etiqueta plausible sacada de un
    contexto que el modelo nunca vio en entrenamiento.
    """
    s = _sesion()
    sujeto = load(1)
    puntos = sujeto.keypoints[sujeto.evaluable]
    for i in range(s.window_frames - 1):
        estado = s.step(feed=(puntos[i], _caja(puntos[i])))
    assert estado["task"] == "esperando contexto", estado["task"]

    estado = s.step(feed=(puntos[s.window_frames - 1], _caja(puntos[s.window_frames - 1])))
    assert estado["task"] != "esperando contexto", "con la ventana llena ya debería predecir"
    assert estado["task"].count("/") == 2, f"formato de tarea inesperado: {estado['task']}"


def test_losing_the_person_clears_the_context() -> None:
    """Si la persona sale de cuadro, la ventana se vacía.

    Mantenerla pegaría el final de una intervención con el principio de la
    siguiente y el modelo vería un movimiento que no ocurrió.
    """
    s = _sesion()
    sujeto = load(1)
    puntos = sujeto.keypoints[sujeto.evaluable]
    for i in range(s.window_frames):
        s.step(feed=(puntos[i], _caja(puntos[i])))
    assert len(s.buffer) == s.window_frames

    estado = s.step(feed=(None, None))
    assert estado["present"] is False and estado["reba"] == 0
    assert len(s.buffer) == 0, "la ventana tenía que vaciarse al perder a la persona"
    assert estado["task"] == "—"


def test_a_sustained_risk_closes_an_event_and_a_brief_one_does_not() -> None:
    """Es la misma regla del modo por lotes, pero cerrando en caliente."""
    s = _sesion(load_kg=18, coupling="poor")
    sujeto = load(1)
    puntos = sujeto.keypoints[sujeto.evaluable]

    # Se busca una postura que en este puesto puntúe por encima del umbral, y se
    # sostiene más de un segundo.
    peligrosa = None
    for p in puntos:
        estado = s.step(feed=(p, _caja(p)))
        if estado["reba"] >= s.config.risk_threshold:
            peligrosa = p
            break
    assert peligrosa is not None, "ninguna postura del sujeto superó el umbral"

    antes = len(s.events)
    for _ in range(int(TARGET_HZ * 1.5)):
        s.step(feed=(peligrosa, _caja(peligrosa)))
    assert len(s.events) == antes, "el evento no se cierra mientras dura"

    segura = min(puntos, key=lambda p: s.step(feed=(p, _caja(p)))["reba"])
    for _ in range(3):
        s.step(feed=(segura, _caja(segura)))
    assert len(s.events) > antes, "al bajar el riesgo tenía que cerrarse el evento"

    evento = s.events[-1]
    assert evento.duration_seconds >= s.config.min_event_seconds
    assert evento.peak_reba >= s.config.risk_threshold
    assert evento.dominant_component


def test_one_flickering_frame_does_not_reset_the_event() -> None:
    """El fallo por el que un video de alguien levantando cajas cerraba CERO eventos.

    No era que no viera el riesgo: 65 de 128 fotogramas pasaban del umbral. El
    puntaje parpadea entre 3 y 4 —"6 4 4 4 3 4 4 3 4 4 3..."— y un solo fotograma por
    debajo reiniciaba la racha, asi que la mas larga duraba 0,9 s contra el 1,0 s que
    se exige. Fallaba por un fotograma.

    Aqui se reproduce exacto: riesgo sostenido, UN fotograma seguro en medio, y mas
    riesgo. Tiene que salir UN evento, no ninguno y no dos.
    """
    s = _sesion(load_kg=18, coupling="poor")
    sujeto = load(1)
    puntos = sujeto.keypoints[sujeto.evaluable]

    peligrosa = None
    for p in puntos:
        if s.step(feed=(p, _caja(p)))["reba"] >= s.config.risk_threshold:
            peligrosa = p
            break
    assert peligrosa is not None
    segura = min(puntos, key=lambda p: s.step(feed=(p, _caja(p)))["reba"])
    assert s.step(feed=(segura, _caja(segura)))["reba"] < s.config.risk_threshold

    antes = len(s.events)
    for _ in range(6):
        s.step(feed=(peligrosa, _caja(peligrosa)))
    s.step(feed=(segura, _caja(segura)))          # el parpadeo
    for _ in range(6):
        s.step(feed=(peligrosa, _caja(peligrosa)))
    assert len(s.events) == antes, "un fotograma suelto no puede cerrar el tramo"

    for _ in range(4):                             # ahora si se baja de verdad
        s.step(feed=(segura, _caja(segura)))
    assert len(s.events) == antes + 1, "tenia que cerrarse UN evento"
    assert s.events[-1].duration_seconds >= 1.0


def test_a_real_pause_still_closes_the_event() -> None:
    """La tolerancia no puede tragarse una pausa de verdad: si no, dos levantamientos
    separados se publicarian como uno solo y su duracion seria falsa."""
    s = _sesion(load_kg=18, coupling="poor")
    sujeto = load(1)
    puntos = sujeto.keypoints[sujeto.evaluable]
    peligrosa = next(
        (p for p in puntos if s.step(feed=(p, _caja(p)))["reba"] >= s.config.risk_threshold),
        None,
    )
    assert peligrosa is not None
    segura = min(puntos, key=lambda p: s.step(feed=(p, _caja(p)))["reba"])

    antes = len(s.events)
    for _ in range(12):
        s.step(feed=(peligrosa, _caja(peligrosa)))
    for _ in range(10):                            # un segundo entero sin riesgo
        s.step(feed=(segura, _caja(segura)))
    assert len(s.events) == antes + 1
    for _ in range(12):
        s.step(feed=(peligrosa, _caja(peligrosa)))
    for _ in range(10):
        s.step(feed=(segura, _caja(segura)))
    assert len(s.events) == antes + 2, "dos tramos separados son dos eventos"


def test_skeleton_sent_to_the_browser_is_in_canvas_range() -> None:
    s = _sesion()
    sujeto = load(1)
    p = sujeto.keypoints[sujeto.evaluable][0]
    estado = s.step(feed=(p, _caja(p)))
    for x, y in estado["skeleton"]:
        assert -0.2 <= x <= 1.2 and -0.2 <= y <= 1.2, f"punto fuera del lienzo: ({x}, {y})"


def test_trust_guard_catches_a_task_that_contradicts_the_geometry() -> None:
    """El guardia que nació de la prueba con la cámara del portátil.

    El modelo contestaba `bend` mientras el tronco medía 2 grados. La tarea y los
    ángulos salen de dos caminos distintos —uno aprendido y otro geométrico— así que
    cuando se contradicen, al menos uno se equivoca, y el geométrico es el que está
    atado a la norma.
    """
    s = _sesion()
    sujeto = load(1)
    erguido = sujeto.keypoints[sujeto.evaluable][0]

    assert s.check_trust("bend / pick-up / low", 2.0, erguido), (
        "no detectó que 'agachado' con 2 grados de tronco es imposible"
    )
    assert s.check_trust("stand / place / mid", 70.0, erguido), (
        "no detectó que 'de pie' con 70 grados de tronco es imposible"
    )
    assert s.check_trust("bend / pick-up / low", 65.0, erguido) is None, (
        "'agachado' con 65 grados de tronco es coherente y no debería avisar"
    )


def test_trust_guard_flags_a_camera_angle_it_never_trained_on() -> None:
    """Fuera del rango de vistas visto, la tarea se marca como no fiable.

    Un modelo que extrapola contesta con la misma seguridad que cuando sabe, y el
    panel lo mostraba igual que cualquier otra predicción.
    """
    s = _sesion()
    sujeto = load(1)
    puntos = sujeto.keypoints[sujeto.evaluable][0].copy()

    # Se separan los hombros a lo ancho hasta salirse del rango entrenado, que es lo
    # que ocurre cuando alguien se pone de frente y cerca de la cámara.
    from src.pose.schema import JOINT

    alto = puntos[:, 1].max() - puntos[:, 1].min()
    centro = (puntos[JOINT["left_shoulder"], 0] + puntos[JOINT["right_shoulder"], 0]) / 2
    exceso = alto * (s.view["shoulder_ratio_p99"] + 0.15)
    puntos[JOINT["left_shoulder"], 0] = centro - exceso / 2
    puntos[JOINT["right_shoulder"], 0] = centro + exceso / 2

    aviso = s.check_trust("bend / pick-up / low", 65.0, puntos)
    assert aviso and "angulo" in aviso, f"no avisó del ángulo fuera de rango: {aviso}"


def test_a_lift_closed_live_carries_its_niosh_analysis() -> None:
    """En vivo, un evento de levantamiento tiene que traer los kilos, como en el panel.

    Y uno que NO sea levantamiento no debe traerlos: la ecuación mide levantar una
    carga, y aplicarla a estar de pie en mala postura daría un índice sin
    significado que el panel mostraría igual de creíble.
    """
    sujeto = load(1)
    puntos = sujeto.keypoints[sujeto.evaluable]

    def correr(tarea: str):
        s = _sesion(load_kg=15, coupling="poor", worker_height_cm=172, lifts_per_min=3)
        peligrosa = next(
            (p for p in puntos if s.step(feed=(p, _caja(p)))["reba"] >= s.config.risk_threshold),
            None,
        )
        assert peligrosa is not None
        # Se sustituye el predictor entero: fijar `_last_task` no vale porque el
        # modelo lo recalcula cada tres fotogramas y lo pisaría.
        s.predict_task = lambda: tarea
        s._last_task = tarea
        for _ in range(int(TARGET_HZ * 1.5)):
            s.step(feed=(peligrosa, _caja(peligrosa)))
        segura = min(puntos, key=lambda p: s.step(feed=(p, _caja(p)))["reba"])
        for _ in range(3):
            s.step(feed=(segura, _caja(segura)))
        return s.events[-1]

    levantando = correr("bend / pick-up / low")
    assert levantando.niosh, "un levantamiento en vivo tiene que traer el analisis NIOSH"
    assert levantando.niosh["recommended_weight_kg"] > 0
    assert levantando.niosh["lifting_index"] is not None, "con peso declarado tiene que haber indice"

    sosteniendo = correr("stand / hold / mid")
    assert sosteniendo.niosh is None, "sostener no es levantar: NIOSH no aplica"


def test_the_weight_applied_is_the_one_seen_not_the_one_configured() -> None:
    """El salto de «una constante del puesto» a «lo que hay en las manos».

    Con catálogo y una carga reconocida, el peso que entra en NIOSH es el de esa
    carga. Sin carga reconocida, se cae al del puesto — y eso también tiene que
    funcionar, porque es lo que pasa cuando el detector no ve nada.
    """
    from src.load.carga import LoadCatalogEntry, build_load

    catalogo = (
        LoadCatalogEntry("pequena", width_cm=28.0, weight_kg=5.0),
        LoadCatalogEntry("grande", width_cm=48.0, weight_kg=18.0),
    )
    s = _sesion(load_kg=12.0, coupling="fair", worker_height_cm=170, load_catalog=catalogo)

    assert s._effective_load_kg == 12.0, "sin carga vista se usa el peso del puesto"

    # Una caja de 48 cm con la escala de una persona de 170 cm en 500 px.
    px_cm = 500 / (170 * 0.94)
    ancho = 48 * px_cm
    s._last_load = build_load(
        [100, 100, 100 + ancho, 180], 0.9, px_cm, catalogo
    )
    assert s._last_load.matched.name == "grande"
    assert s._effective_load_kg == 18.0, (
        "con la carga reconocida, el peso aplicado es el suyo y no el del puesto"
    )


def test_live_payload_has_no_image_unless_preview_is_asked() -> None:
    """La imagen solo viaja cuando se pide la vista de instalación, y aun así es
    efímera: ningún camino del código la escribe en disco."""
    s = _sesion()
    sujeto = load(1)
    p = sujeto.keypoints[sujeto.evaluable][0]
    estado = s.step(feed=(p, _caja(p)))
    assert "preview" not in estado


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
