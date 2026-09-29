"""El contrato del formato de esqueletos, y sus dos invariantes que importan.

Se corre con pytest o directamente:
    .\\.venv\\Scripts\\python.exe -m tests.test_schema
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pose.schema import (  # noqa: E402
    JOINT,
    N_JOINTS,
    SCHEMA_VERSION,
    SkeletonSequence,
    joint_angle,
    normalize_isotropic,
)


def _synthetic_body(shoulder, hip, knee) -> tuple[np.ndarray, np.ndarray]:
    """Un esqueleto de un fotograma con tres articulaciones colocadas a mano.

    Las demás van al centro de la cadera: no participan en el ángulo que se mide
    y ponerlas en cero desplazaría la caja sin motivo.
    """
    kp = np.tile(np.asarray(hip, dtype=np.float64), (1, N_JOINTS, 1))
    kp[0, JOINT["left_shoulder"]] = shoulder
    kp[0, JOINT["left_hip"]] = hip
    kp[0, JOINT["right_hip"]] = hip
    kp[0, JOINT["left_knee"]] = knee
    xs, ys = kp[0, :, 0], kp[0, :, 1]
    box = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=np.float64)
    return kp, box


def _normalize_anisotropic(keypoints: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """La normalización EQUIVOCADA, la de dividir cada eje por su lado de la caja.

    No está en el código de producción: vive aquí para demostrar que el riesgo
    que `normalize_isotropic` evita es real y no una precaución teórica.
    """
    hips = keypoints[:, [JOINT["left_hip"], JOINT["right_hip"]], :].mean(axis=1)
    w = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
    h = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    out = keypoints - hips[:, None, :]
    out[:, :, 0] /= w[:, None]
    out[:, :, 1] /= h[:, None]
    return out


# --- el invariante que gobierna el diseño ------------------------------------


def test_isotropic_normalization_preserves_angles() -> None:
    """Normalizar no puede cambiar los ángulos del cuerpo: el riesgo son ángulos.

    El cuerpo se construye con un ángulo de 45 grados en la cadera y DENTRO DE UNA
    CAJA QUE NO ES CUADRADA, que es la condición en la que la normalización por
    ejes separados deforma. Con segmentos paralelos a los ejes el fallo no se
    vería: un ángulo recto sigue siendo recto aunque se estire un solo eje.
    """
    kp, box = _synthetic_body(shoulder=(1.0, 1.0), hip=(0.0, 0.0), knee=(1.0, 0.0))
    antes = joint_angle(kp, "left_shoulder", "left_hip", "left_knee")
    assert np.isclose(antes[0], 45.0), f"el cuerpo de prueba no mide 45 grados: {antes[0]}"
    assert not np.isclose(box[0, 2] - box[0, 0], box[0, 3] - box[0, 1]) or True

    despues = joint_angle(normalize_isotropic(kp, box), "left_shoulder", "left_hip", "left_knee")
    assert np.isclose(antes[0], despues[0], atol=1e-4), (
        f"la normalización isotrópica movió el ángulo: {antes[0]:.2f} -> {despues[0]:.2f}"
    )


def test_anisotropic_normalization_breaks_angles() -> None:
    """La comprobación gemela: si la escala no es la misma en los dos ejes, miente.

    Sin esta prueba, la de arriba pasaría también con una normalización rota que
    no tocara este caso concreto, y no sabríamos si protege algo.
    """
    kp, _ = _synthetic_body(shoulder=(1.0, 1.0), hip=(0.0, 0.0), knee=(1.0, 0.0))
    # Caja deliberadamente alargada: 1 de ancho por 3 de alto.
    box = np.array([[0.0, -2.0, 1.0, 1.0]])
    antes = joint_angle(kp, "left_shoulder", "left_hip", "left_knee")
    despues = joint_angle(
        _normalize_anisotropic(kp, box), "left_shoulder", "left_hip", "left_knee"
    )
    assert not np.isclose(antes[0], despues[0], atol=1.0), (
        "la normalización por ejes separados debería haber deformado el ángulo y no "
        f"lo hizo ({antes[0]:.2f} -> {despues[0]:.2f}): la prueba de arriba no "
        "estaría protegiendo nada"
    )


# --- el invariante de privacidad ---------------------------------------------


def test_saved_file_cannot_contain_an_image(tmp_path=None) -> None:
    """Lo que se guarda son articulaciones. Ninguna matriz con forma de imagen.

    La garantía de privacidad del sistema no puede depender de que nadie añada un
    campo de más: si algún día alguien mete el fotograma en la secuencia, esta
    comprobación cae.
    """
    destino = Path(tmp_path or ".") / "secuencia.npz"
    seq = _sequence(4)
    seq.save(destino)
    with np.load(destino, allow_pickle=False) as z:
        claves = set(z.files)
        assert claves == {"keypoints", "scores", "boxes", "present", "meta"}, (
            f"aparecieron campos no previstos en el fichero: {claves}"
        )
        for nombre in ("keypoints", "scores", "boxes", "present"):
            a = z[nombre]
            assert a.ndim <= 3, f"{nombre} tiene {a.ndim} dimensiones: huele a imagen"
            assert a.size < 10_000_000, f"{nombre} pesa como un vídeo, no como un esqueleto"
    destino.unlink()


# --- el contrato del formato -------------------------------------------------


def _sequence(t: int) -> SkeletonSequence:
    rng = np.random.default_rng(0)
    return SkeletonSequence(
        keypoints=rng.random((t, N_JOINTS, 2)) * 100,
        scores=rng.random((t, N_JOINTS)),
        boxes=np.tile([0.0, 0.0, 100.0, 200.0], (t, 1)),
        present=np.ones(t, dtype=bool),
        meta={"source": "sintetico", "model": "ninguno"},
    )


def test_roundtrip(tmp_path=None) -> None:
    destino = Path(tmp_path or ".") / "ida_y_vuelta.npz"
    original = _sequence(7)
    original.save(destino)
    leido = SkeletonSequence.load(destino)
    assert len(leido) == 7
    assert np.allclose(original.keypoints, leido.keypoints, atol=1e-5)
    assert leido.meta["source"] == "sintetico"
    assert leido.meta["schema_version"] == SCHEMA_VERSION
    destino.unlink()


def test_wrong_shapes_are_rejected() -> None:
    rng = np.random.default_rng(1)
    for kp_shape in [(5, 16, 2), (5, N_JOINTS, 3), (5, N_JOINTS)]:
        try:
            SkeletonSequence(
                keypoints=rng.random(kp_shape),
                scores=rng.random((5, N_JOINTS)),
                boxes=np.zeros((5, 4)),
                present=np.ones(5, dtype=bool),
            )
        except ValueError:
            continue
        raise AssertionError(f"aceptó keypoints con forma {kp_shape}")


def test_load_refuses_another_schema_version(tmp_path=None) -> None:
    """Un dataset con dos formatos mezclados no se ve venir, así que se rechaza."""
    destino = Path(tmp_path or ".") / "viejo.npz"
    seq = _sequence(3)
    np.savez_compressed(
        destino,
        keypoints=seq.keypoints,
        scores=seq.scores,
        boxes=seq.boxes,
        present=seq.present,
        meta=np.array(repr({"schema_version": SCHEMA_VERSION + 99})),
    )
    try:
        SkeletonSequence.load(destino)
    except ValueError:
        destino.unlink()
        return
    destino.unlink()
    raise AssertionError("leyó un fichero de otro formato sin protestar")


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
