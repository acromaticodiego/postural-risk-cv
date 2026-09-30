"""De vídeo a esqueletos. Es la única pieza del sistema que ve imágenes.

Y es donde vive la garantía de privacidad: los fotogramas entran, se extraen las
articulaciones y el fotograma se descarta en la misma iteración. Ninguna función
de este módulo devuelve, guarda ni acumula imágenes, así que ningún consumidor
puede persistir una aunque quiera. Esa es la diferencia entre una garantía y una
promesa del README.

Tampoco se remuestrea ni se normaliza nada aquí: lo que sale son píxeles crudos a
la tasa original del vídeo, con la procedencia anotada. Cocinar los datos en la
extracción obliga a volver a pasar horas de vídeo por la GPU cada vez que se
cambia de idea sobre el preprocesado.

Uso:
    .\\.venv\\Scripts\\python.exe -m src.pose.extractor VIDEO --out salida.npz
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from .schema import N_JOINTS, SkeletonSequence

DEFAULT_MODEL = "yolo11n-pose.pt"


def extract_tracks(
    video_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    device: int | str = 0,
    stride: int = 1,
    max_frames: int | None = None,
) -> dict[int, SkeletonSequence]:
    """Devuelve una secuencia de esqueletos por persona seguida en el vídeo.

    `stride` procesa uno de cada N fotogramas. Queda anotado en la procedencia
    junto al fps efectivo, porque una secuencia submuestreada no es comparable con
    una completa y eso tiene que poder comprobarse después, no recordarse.

    Los huecos son explícitos: en los fotogramas donde el seguidor pierde a una
    persona, `present` queda en False y sus articulaciones en cero. No se
    interpola, porque un hueco tapado es una postura que nadie adoptó.
    """
    from ultralytics import YOLO

    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"no se pudo abrir el vídeo: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = YOLO(model_name)

    # Por track: índice del fotograma procesado -> (keypoints, scores, box)
    per_track: dict[int, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    processed = 0
    read_index = 0
    started = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if read_index % stride != 0:
            read_index += 1
            continue
        read_index += 1

        results = model.track(frame, persist=True, device=device, verbose=False)[0]
        # El fotograma deja de existir aquí. Nada de lo que sigue lo referencia.
        del frame

        if results.boxes is not None and results.boxes.id is not None:
            ids = results.boxes.id.int().cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy()
            kp = results.keypoints.data.cpu().numpy()  # (n, 17, 3): x, y, conf
            for slot, track_id in enumerate(ids):
                per_track.setdefault(int(track_id), {})[processed] = (
                    kp[slot, :, :2],
                    kp[slot, :, 2],
                    boxes[slot],
                )
        processed += 1
        if max_frames is not None and processed >= max_frames:
            break

    capture.release()
    elapsed = time.perf_counter() - started

    meta_base = {
        "source": video_path.name,
        "model": model_name,
        "device": str(device),
        "source_fps": round(source_fps, 3),
        "stride": stride,
        "effective_fps": round(source_fps / stride, 3) if source_fps else None,
        "resolution": [width, height],
        "frames_processed": processed,
        "extracted_on": date.today().isoformat(),
        "extraction_seconds": round(elapsed, 2),
    }

    sequences: dict[int, SkeletonSequence] = {}
    for track_id, frames in per_track.items():
        keypoints = np.zeros((processed, N_JOINTS, 2), dtype=np.float32)
        scores = np.zeros((processed, N_JOINTS), dtype=np.float32)
        boxes_arr = np.zeros((processed, 4), dtype=np.float32)
        present = np.zeros(processed, dtype=bool)
        for index, (kp, sc, box) in frames.items():
            keypoints[index] = kp
            scores[index] = sc
            boxes_arr[index] = box
            present[index] = True
        sequences[track_id] = SkeletonSequence(
            keypoints=keypoints,
            scores=scores,
            boxes=boxes_arr,
            present=present,
            meta=meta_base | {"track_id": track_id, "frames_present": int(present.sum())},
        )
    return sequences


def extract_single_subject(
    video_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    device: int | str = 0,
    stride: int = 1,
    max_frames: int | None = None,
) -> SkeletonSequence:
    """Para vídeos con UNA sola persona: detecta por fotograma y no sigue a nadie.

    Existe porque el seguimiento hace daño cuando no hace falta. Medido el 29/09
    sobre el sujeto 01 de UW-IOM, que tiene un único participante: `extract_tracks`
    produjo 14 identidades distintas, y la mayor cubría 515 de 1474 fotogramas
    mientras otra de 500 era la misma persona partida por la mitad. Quedarse con
    una sola identidad tiraba el 65% del vídeo, y la tirada no era aleatoria: se
    perdían los tramos donde el detector titubea, que son justamente los de las
    posturas raras que este proyecto quiere medir.

    El criterio por fotograma es la caja de mayor ÁREA. En estos vídeos el
    participante está cerca de la cámara y cualquier otra detección es de fondo. No
    vale para una planta con varios operarios: allí hay que seguir e identificar, y
    a quién se mide pasa a ser una decisión de producto.
    """
    from ultralytics import YOLO

    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"no se pudo abrir el vídeo: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = YOLO(model_name)

    keypoints: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    present: list[bool] = []
    read_index = 0
    started = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if read_index % stride != 0:
            read_index += 1
            continue
        read_index += 1

        result = model.predict(frame, device=device, verbose=False)[0]
        del frame  # el fotograma deja de existir aquí

        boxes_xyxy = None if result.boxes is None else result.boxes.xyxy.cpu().numpy()
        if boxes_xyxy is not None and len(boxes_xyxy):
            areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
            slot = int(areas.argmax())
            kp = result.keypoints.data.cpu().numpy()[slot]
            keypoints.append(kp[:, :2])
            scores.append(kp[:, 2])
            boxes.append(boxes_xyxy[slot])
            present.append(True)
        else:
            keypoints.append(np.zeros((N_JOINTS, 2), dtype=np.float32))
            scores.append(np.zeros(N_JOINTS, dtype=np.float32))
            boxes.append(np.zeros(4, dtype=np.float32))
            present.append(False)

        if max_frames is not None and len(present) >= max_frames:
            break

    capture.release()
    present_arr = np.array(present, dtype=bool)
    return SkeletonSequence(
        keypoints=np.asarray(keypoints, dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
        boxes=np.asarray(boxes, dtype=np.float32),
        present=present_arr,
        meta={
            "source": video_path.name,
            "model": model_name,
            "device": str(device),
            "source_fps": round(source_fps, 3),
            "stride": stride,
            "effective_fps": round(source_fps / stride, 3) if source_fps else None,
            "resolution": [width, height],
            "frames_processed": len(present),
            "frames_present": int(present_arr.sum()),
            "selection": "largest_box_per_frame",
            "tracking": False,
            "extracted_on": date.today().isoformat(),
            "extraction_seconds": round(time.perf_counter() - started, 2),
        },
    )


def main_subject(sequences: dict[int, SkeletonSequence]) -> SkeletonSequence:
    """La persona con más fotogramas presentes.

    El criterio se elige así, y no por área de la caja, porque el área premia a
    quien pasa cerca de la cámara un instante. En los vídeos de un solo
    participante da igual; en cuanto haya dos personas en escena este criterio deja
    de valer y hará falta decidir a quién se mide, que es una decisión de producto.
    """
    if not sequences:
        raise ValueError("no se encontró ninguna persona en el vídeo")
    return max(sequences.values(), key=lambda s: int(s.present.sum()))


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Extrae esqueletos de un vídeo.")
    parser.add_argument("video")
    parser.add_argument("--out", default=None, help="fichero .npz de salida")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="0")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    device: int | str = int(args.device) if args.device.isdigit() else args.device
    sequences = extract_tracks(
        args.video,
        model_name=args.model,
        device=device,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    print(f"personas seguidas: {len(sequences)}")
    for track_id, seq in sorted(sequences.items(), key=lambda kv: -int(kv[1].present.sum())):
        print(f"  track {track_id:>3}: {int(seq.present.sum()):>5} de {len(seq)} fotogramas")

    principal = main_subject(sequences)
    meta = principal.meta
    print(
        f"\nprocedencia: {meta['source']} | {meta['model']} | "
        f"{meta['source_fps']} fps | paso {meta['stride']} | "
        f"{meta['frames_processed']} fotogramas en {meta['extraction_seconds']} s"
    )
    if args.out:
        destino = Path(args.out)
        destino.parent.mkdir(parents=True, exist_ok=True)
        principal.save(destino)
        print(f"guardado: {destino} ({destino.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    _cli()
