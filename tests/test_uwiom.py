"""Que el dataset esté bien emparejado, comprobado sobre los 20 sujetos.

La prueba que importa es la de coherencia física: si la alineación entre
esqueletos y etiquetas se rompiera, nada daría un error —los arrays seguirían
teniendo el tamaño correcto— y el modelo entrenaría sobre un dataset desplazado.
Lo único que lo delata es que las etiquetas dejen de describir lo que el cuerpo
hace.

    .\\.venv\\Scripts\\python.exe -m tests.test_uwiom
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.uwiom import (  # noqa: E402
    KINECT_BROKEN,
    LABEL_FIELDS,
    SUBJECTS,
    load_subject,
    parse_label,
    read_kinect_skeleton,
)
from src.pose.schema import JOINT  # noqa: E402

# Vocabulario observado en los ficheros de etiquetas. Un valor nuevo no es un
# fallo del dataset, es una señal de que hay que mirarlo antes de entrenar.
VOCABULARY = {
    "object": {"box", "rod", "none"},
    "motion": {"walk", "stand", "bend"},
    "manipulation": {"pick-up", "place", "hold", "reach", "none"},
    "height": {"low", "mid", "top", "none"},
}

MIN_SEPARATION_DEGREES = 10.0


def trunk_inclination(keypoints: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Grados del tronco respecto a la vertical de la imagen, con NaN donde falta.

    La y de una imagen crece hacia abajo, así que la vertical de referencia es -y:
    de pie, el tronco apunta hacia arriba y el ángulo sale cerca de cero.
    """
    hip = keypoints[:, [JOINT["left_hip"], JOINT["right_hip"]], :].mean(axis=1)
    shoulder = keypoints[:, [JOINT["left_shoulder"], JOINT["right_shoulder"]], :].mean(axis=1)
    trunk = shoulder - hip
    norms = np.linalg.norm(trunk, axis=1)
    cos = np.divide(-trunk[:, 1], norms, out=np.zeros(len(trunk)), where=norms > 1e-9)
    return np.where(present, np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))), np.nan)


# --- la prueba que sostiene el dataset ---------------------------------------


def test_alignment_is_physically_coherent() -> None:
    """En los fotogramas `bend` el tronco tiene que estar más inclinado que en `stand`.

    Se exige en LOS VEINTE sujetos y con margen, no de media: una alineación rota
    en un solo sujeto mete ~1.500 fotogramas mal etiquetados, y promediar lo
    esconde.
    """
    fallos = []
    for i in SUBJECTS:
        s = load_subject(i)
        inclinacion = trunk_inclination(s.skeleton.keypoints, s.skeleton.present)
        motion = s.field("motion")
        bend = inclinacion[motion == "bend"]
        stand = inclinacion[motion == "stand"]
        bend, stand = bend[~np.isnan(bend)], stand[~np.isnan(stand)]
        if len(bend) < 10 or len(stand) < 10:
            fallos.append(f"sujeto {i}: solo {len(bend)} bend y {len(stand)} stand")
            continue
        separacion = bend.mean() - stand.mean()
        if separacion < MIN_SEPARATION_DEGREES:
            fallos.append(f"sujeto {i}: separación {separacion:+.2f}°")
    assert not fallos, (
        "las etiquetas no describen la postura en estos sujetos, así que la "
        "alineación está mal: " + "; ".join(fallos)
    )


def test_real_fps_is_not_the_container_fps() -> None:
    """La tasa real sale de las marcas de tiempo, no de la cabecera del vídeo.

    Protege un hallazgo del 29/09: la cabecera dice 8, 10, 11 o 12 redondos y la
    captura real va de 7,81 a 10,58. Si alguien 'simplifica' el adaptador para
    leer el fps del contenedor, esta prueba cae.
    """
    diferencias = []
    for i in SUBJECTS:
        s = load_subject(i)
        contenedor = s.skeleton.meta.get("container_fps")
        assert 6.0 < s.fps < 13.0, f"sujeto {i}: fps real {s.fps:.2f}, fuera de lo esperable"
        if contenedor:
            diferencias.append(abs(s.fps - contenedor))
    assert max(diferencias) > 0.5, (
        "el fps del contenedor coincide con el real en todos los sujetos, así que "
        "este hallazgo ya no describe el material y hay que revisarlo"
    )


def test_counts_match_across_sources() -> None:
    for i in SUBJECTS:
        s = load_subject(i)
        assert len(s.skeleton) == len(s.labels) == len(s.timestamps)
        assert s.skeleton.meta["label_offset"] >= 0


def test_labels_use_the_known_vocabulary() -> None:
    desconocidos = []
    for i in SUBJECTS:
        s = load_subject(i)
        for campo in LABEL_FIELDS:
            vistos = set(s.field(campo))
            extra = vistos - VOCABULARY[campo]
            if extra:
                desconocidos.append(f"sujeto {i}, {campo}: {extra}")
    assert not desconocidos, "valores de etiqueta nunca vistos: " + "; ".join(desconocidos)


def test_malformed_label_is_rejected() -> None:
    for linea in ["box_bend_pick-up", "box_bend_pick-up_low_extra", ""]:
        try:
            parse_label(linea)
        except ValueError:
            continue
        raise AssertionError(f"aceptó la etiqueta mal formada {linea!r}")


def test_broken_kinect_subject_is_refused() -> None:
    """Pedir el esqueleto de Kinect de un sujeto defectuoso tiene que fallar.

    El sujeto 3 da una separación bend/stand negativa con el Kinect, o sea que su
    esqueleto 3D no describe lo que hace el cuerpo. Su vídeo sí, y por eso el
    sujeto se conserva; lo que no puede pasar es que alguien lo use sin saberlo.
    """
    assert KINECT_BROKEN, "si no hay sujetos marcados, esta prueba no comprueba nada"
    for i in KINECT_BROKEN:
        try:
            read_kinect_skeleton(i)
        except ValueError:
            continue
        raise AssertionError(f"devolvió el esqueleto de Kinect del sujeto {i}, que está roto")


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
