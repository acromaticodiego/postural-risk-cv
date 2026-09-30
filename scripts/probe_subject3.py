"""¿Por qué el sujeto 3 de UW-IOM no cuadra?

Con la alineación al final —que gana en 18 de los 20 sujetos— el sujeto 3 da una
separación bend/stand de -2,67 grados: sus fotogramas etiquetados `bend` salen
MENOS inclinados que los `stand`, que es imposible si las etiquetas y el esqueleto
se corresponden.

DOS HIPÓTESIS, y esta sonda las separa:

  H1. El esqueleto del Kinect de ese sujeto está mal —el sensor perdió a la
      persona, o siguió otra cosa—. Entonces, extrayendo el esqueleto del VÍDEO
      con YOLO, la separación tiene que salir positiva y normal.
  H2. Las etiquetas de ese sujeto están mal, o se alinean de otra forma. Entonces
      con YOLO saldrá igual de mal, porque el problema no está en el esqueleto.

EL CONTROL, sin el cual esto no diagnostica nada: se hace lo mismo con un sujeto
que sí cuadra. Si YOLO reprodujera valores raros también ahí, el resultado del
sujeto 3 no diría nada sobre el sujeto 3, sino sobre mi cálculo con YOLO.

    .\\.venv\\Scripts\\python.exe scripts\\probe_subject3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pose.extractor import extract_single_subject  # noqa: E402
from src.pose.schema import JOINT  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
BASE = RAIZ / "data/raw/uw-iom/xwzzkxtf9s-2/UW IOM Dataset"
VIDEOS = RAIZ / "data/raw/uw-iom/videos"

SPINE_BASE, SPINE_SHOULDER = 0, 20


def labels_of(index: int) -> list[str]:
    return [
        line.strip()
        for line in (BASE / "VideoLabels" / f"{index:02d}.txt")
        .read_text(encoding="utf-8", errors="replace")
        .split("\n")
        if line.strip()
    ]


def inclination_kinect(index: int) -> tuple[np.ndarray, int]:
    with h5py.File(BASE / "JointPositions" / f"{index}.mat", "r") as f:
        joints = np.array(f["bodylogger3D"])
        n_video = f["videotimelogger"].shape[0]
    trunk = joints[:, SPINE_SHOULDER, :] - joints[:, SPINE_BASE, :]
    norms = np.linalg.norm(trunk, axis=1)
    cos = np.divide(trunk[:, 1], norms, out=np.zeros(len(trunk)), where=norms > 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))), n_video


def inclination_yolo(keypoints: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Igual que en el Kinect pero con articulaciones COCO y en coordenadas de imagen.

    El tronco va del punto medio de las caderas al punto medio de los hombros. En
    una imagen la y crece HACIA ABAJO, así que la vertical de referencia es -y: de
    pie, el tronco apunta hacia arriba y el ángulo sale cerca de cero.
    """
    hip = keypoints[:, [JOINT["left_hip"], JOINT["right_hip"]], :].mean(axis=1)
    shoulder = keypoints[:, [JOINT["left_shoulder"], JOINT["right_shoulder"]], :].mean(axis=1)
    trunk = shoulder - hip
    norms = np.linalg.norm(trunk, axis=1)
    cos = np.divide(-trunk[:, 1], norms, out=np.zeros(len(trunk)), where=norms > 1e-9)
    grados = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.where(present, grados, np.nan)


def separation(values: np.ndarray, labels: list[str]) -> tuple[float, int, int]:
    is_bend = np.array(["_bend_" in f"_{lab}_" for lab in labels])
    is_stand = np.array(["_stand_" in f"_{lab}_" for lab in labels])
    bend, stand = values[is_bend], values[is_stand]
    bend, stand = bend[~np.isnan(bend)], stand[~np.isnan(stand)]
    if len(bend) < 10 or len(stand) < 10:
        return float("nan"), len(bend), len(stand)
    return float(bend.mean() - stand.mean()), len(bend), len(stand)


def diagnose(index: int) -> None:
    print(f"\n{'=' * 66}\nSUJETO {index:02d}\n{'=' * 66}")
    labels = labels_of(index)
    inc_k, n_video = inclination_kinect(index)

    hueco = len(inc_k) - len(labels)
    sep_k, n_bend, n_stand = separation(inc_k[hueco : hueco + len(labels)], labels)
    print(f"  etiquetas {len(labels)} | esqueletos Kinect {len(inc_k)} | vídeo {n_video}")
    print(f"  fotogramas bend {n_bend}, stand {n_stand}")
    print(f"  KINECT, alineado al final: separación {sep_k:+.2f}°")

    video = VIDEOS / f"{index:02d}.avi"
    if not video.exists():
        print(f"  (falta {video.name}: no se puede comparar con YOLO)")
        return

    seq = extract_single_subject(video, device=0)
    print(
        f"  YOLO sin seguimiento: {seq.meta['frames_present']} de "
        f"{seq.meta['frames_processed']} fotogramas con persona "
        f"({seq.meta['extraction_seconds']} s)"
    )
    inc_y = inclination_yolo(seq.keypoints, seq.present)

    hueco_y = len(inc_y) - len(labels)
    if hueco_y < 0:
        print(f"  YOLO dio {len(inc_y)} fotogramas, menos que las {len(labels)} etiquetas")
        return
    sep_y, n_bend_y, n_stand_y = separation(inc_y[hueco_y : hueco_y + len(labels)], labels)
    visibles = int(np.sum(~np.isnan(inc_y)))
    print(
        f"  YOLO, alineado al final: separación {sep_y:+.2f}°  "
        f"({visibles} de {len(inc_y)} fotogramas con persona)"
    )
    print(f"  bend con esqueleto {n_bend_y}, stand {n_stand_y}")


def main() -> None:
    print(__doc__.split("\n\n")[0])
    # El control primero, a propósito: si el control sale mal, el resultado del
    # sujeto 3 no se puede interpretar y hay que parar aquí.
    diagnose(1)
    diagnose(3)
    print(
        "\nLECTURA: si en el sujeto 1 las dos columnas se parecen y en el 3 solo "
        "YOLO sale positiva, el esqueleto del Kinect de ese sujeto es el que está "
        "mal (H1). Si en el 3 las dos salen mal, el problema son sus etiquetas (H2)."
    )


if __name__ == "__main__":
    main()
