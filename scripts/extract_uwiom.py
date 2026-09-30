"""Extrae los esqueletos de los 20 sujetos de UW-IOM, del vídeo RGB.

Es la señal que habrá en producción: una cámara normal, sin sensor de profundidad.
El esqueleto 3D del Kinect que trae el dataset se conserva aparte como referencia,
para poder medir cuánto se pierde al bajar de sensor de profundidad a cámara.

Descomprime, extrae y BORRA el `.avi` de cada sujeto antes de pasar al siguiente:
los veinte vídeos descomprimidos son unos 13 GB y no hacen falta a la vez. Lo que
queda son los `.npz` de esqueletos, que pesan megabytes.

Es idempotente: un sujeto que ya tenga su `.npz` se salta, salvo `--rehacer`.

    .\\.venv\\Scripts\\python.exe scripts\\extract_uwiom.py
    .\\.venv\\Scripts\\python.exe scripts\\extract_uwiom.py --solo 3 --rehacer
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pose.extractor import extract_single_subject  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVOS = RAIZ / "data/raw/uw-iom/xwzzkxtf9s-2/UW IOM Dataset/Videos"
VIDEOS = RAIZ / "data/raw/uw-iom/videos"
SALIDA = RAIZ / "data/keypoints/uwiom"


def unpack(index: int) -> Path:
    import py7zr

    destino = VIDEOS / f"{index:02d}.avi"
    if destino.exists():
        return destino
    VIDEOS.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(ARCHIVOS / f"{index:02d}.7z", "r") as z:
        z.extractall(path=VIDEOS)
    if not destino.exists():
        raise FileNotFoundError(f"{index:02d}.7z no contenía {destino.name}")
    return destino


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae esqueletos de UW-IOM.")
    parser.add_argument("--solo", type=int, default=None, help="un solo sujeto")
    parser.add_argument("--rehacer", action="store_true", help="rehacer los ya extraídos")
    parser.add_argument("--conservar-videos", action="store_true")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    SALIDA.mkdir(parents=True, exist_ok=True)
    sujetos = [args.solo] if args.solo else list(range(1, 21))
    device: int | str = int(args.device) if args.device.isdigit() else args.device

    total_t = time.perf_counter()
    print("suj  fotogramas  con persona   %   segundos   fps del vídeo")
    for i in sujetos:
        destino_npz = SALIDA / f"{i:02d}.npz"
        if destino_npz.exists() and not args.rehacer:
            print(f"{i:3d}  ya extraído, se salta (--rehacer para forzar)")
            continue

        video = unpack(i)
        seq = extract_single_subject(video, device=device)
        seq.save(destino_npz)

        meta = seq.meta
        presentes = meta["frames_present"]
        total = meta["frames_processed"]
        print(
            f"{i:3d}  {total:10d}  {presentes:11d}  {100 * presentes / total:5.1f}  "
            f"{meta['extraction_seconds']:8.1f}   {meta['source_fps']:.2f}"
        )
        if not args.conservar_videos:
            video.unlink()

    print(f"\nterminado en {(time.perf_counter() - total_t) / 60:.1f} min")
    pesos = sum(p.stat().st_size for p in SALIDA.glob("*.npz"))
    print(f"esqueletos de {len(list(SALIDA.glob('*.npz')))} sujetos: {pesos / 1e6:.1f} MB en total")


if __name__ == "__main__":
    main()
