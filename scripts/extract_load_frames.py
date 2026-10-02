"""Sacar de un vídeo los fotogramas con los que se afina el detector de carga.

ESTE GUION ESCRIBE IMÁGENES AL DISCO, y es el único del proyecto que lo hace. No
contradice la garantía de privacidad: la garantía es sobre el sistema que se
instala en una planta —`src/pose/extractor.py` no tiene forma de guardar un
fotograma, y por ahí pasa todo lo que el producto procesa—. Esto es una herramienta
de ETIQUETADO, que se corre a mano, sobre vídeos propios y una sola vez por cliente
para calibrar el modelo con sus cargas. Escribe bajo `data/`, que no entra al
repositorio.

QUÉ SACA, Y POR QUÉ NO ELIGE AQUÍ. Saca TODOS los fotogramas donde hay una persona
con las dos muñecas a la vista, que es la condición mínima para poder pre-etiquetar.
La selección de un conjunto variado se hace después, en `prelabel_sam.py`, y el orden
no es un detalle: elegir aquí por variedad de postura llenaba el conjunto de
fotogramas sin ninguna carga en las manos —medido sobre un vídeo real, 11 de 12
elegidos no tenían nada que segmentar—. La variedad hay que buscarla entre los
fotogramas donde SÍ hay carga, y eso no se sabe hasta que SAM lo ha intentado.

Y LA PROCEDENCIA SE DECLARA, NO SE ADIVINA. `--fuente` es obligatorio y dice si el
vídeo es de una cámara o una captura de pantalla. Una captura de pantalla del panel
lleva el esqueleto PINTADO ENCIMA de la persona y de la carga, así que afinar con
ella enseñaría al modelo a buscar líneas de colores; `finetune_load.py` se niega a
entrenar con material declarado `pantalla`. Se podría intentar detectarlo mirando
los píxeles, y se descartó: un test de color confundiría una planta verde con un
trazo del panel, y de qué cámara salió un vídeo es un dato que quien lo grabó
conoce. Es el mismo criterio del ADR 0001 — lo que se sabe se declara, no se
estima.

    .\\.venv\\Scripts\\python.exe scripts\\extract_load_frames.py ^
        --video "C:\\ruta\\carga1.mp4" --fuente camara --por-video 40
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.load.prelabel import MIN_WRIST_SCORE, pick_varied, pose_vector  # noqa: E402
from src.pose.schema import JOINT  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
SALIDA_POR_DEFECTO = RAIZ / "data/carga"
MODELO_POSE = "yolo11n-pose.pt"

# Confianza mínima de la caja de la persona. Por debajo de esto el esqueleto no es de
# fiar y los puntos de aviso para SAM saldrían de coordenadas inventadas.
MIN_PERSON_CONF = 0.40


def _recorte(texto: str | None) -> tuple[float, float, float, float] | None:
    """`x0,y0,x1,y1` en FRACCIONES del fotograma, para que no dependa de la pantalla."""
    if not texto:
        return None
    partes = [float(v) for v in texto.split(",")]
    if len(partes) != 4 or not all(0.0 <= v <= 1.0 for v in partes):
        raise ValueError("el recorte son cuatro fracciones entre 0 y 1: x0,y0,x1,y1")
    return tuple(partes)  # type: ignore[return-value]


def candidatos(video: Path, device: int | str, recorte, paso: int):
    """Los fotogramas donde hay una persona con las dos muñecas a la vista.

    Se exigen las muñecas aquí y no después porque es el requisito que define si el
    fotograma sirve: sin manos localizadas no hay dónde poner el punto positivo de
    SAM, y un fotograma sin aviso no se puede pre-etiquetar.
    """
    from ultralytics import YOLO

    captura = cv2.VideoCapture(str(video))
    if not captura.isOpened():
        raise FileNotFoundError(f"no se pudo abrir el vídeo: {video}")
    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
    modelo = YOLO(MODELO_POSE)

    salida = []
    indice = 0
    descartes = {"sin_persona": 0, "sin_munecas": 0}
    while True:
        ok, fotograma = captura.read()
        if not ok:
            break
        if indice % paso != 0:
            indice += 1
            continue
        actual = indice
        indice += 1

        if recorte:
            alto, ancho = fotograma.shape[:2]
            x0, y0, x1, y1 = recorte
            fotograma = fotograma[int(y0 * alto) : int(y1 * alto), int(x0 * ancho) : int(x1 * ancho)]

        resultado = modelo.predict(fotograma, device=device, conf=MIN_PERSON_CONF, verbose=False)[0]
        if resultado.boxes is None or not len(resultado.boxes):
            descartes["sin_persona"] += 1
            continue
        cajas = resultado.boxes.xyxy.cpu().numpy()
        areas = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
        i = int(areas.argmax())
        datos = resultado.keypoints.data.cpu().numpy()[i]
        kp, scores = datos[:, :2], datos[:, 2]

        if min(scores[JOINT["left_wrist"]], scores[JOINT["right_wrist"]]) < MIN_WRIST_SCORE:
            descartes["sin_munecas"] += 1
            continue

        salida.append(
            {
                "indice": actual,
                "t_s": round(actual / fps, 3),
                "keypoints": kp.round(2).tolist(),
                "scores": scores.round(3).tolist(),
                "box": cajas[i].round(2).tolist(),
                "imagen": fotograma.copy(),
            }
        )

    captura.release()
    return salida, descartes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", required=True, help="repetible")
    parser.add_argument(
        "--fuente",
        required=True,
        choices=("camara", "pantalla"),
        help="de dónde salió el vídeo. 'pantalla' no sirve para afinar y queda marcado",
    )
    parser.add_argument("--salida", default=str(SALIDA_POR_DEFECTO))
    parser.add_argument(
        "--tope-por-video",
        type=int,
        default=400,
        help="tope de seguridad; si se pasa, se quedan los más variados en postura",
    )
    parser.add_argument("--min-separacion-s", type=float, default=0.2)
    parser.add_argument("--paso", type=int, default=5, help="examina uno de cada N fotogramas")
    parser.add_argument("--recorte", default=None, help="x0,y0,x1,y1 en fracciones")
    parser.add_argument("--device", default=0)
    args = parser.parse_args()

    recorte = _recorte(args.recorte)
    salida = Path(args.salida)
    carpeta_frames = salida / "frames"
    carpeta_frames.mkdir(parents=True, exist_ok=True)

    if args.fuente == "pantalla":
        print("AVISO: material declarado como captura de pantalla.")
        print("       Sirve para ensayar el flujo; NO sirve para afinar el modelo.\n")

    manifiesto = {
        "creado": date.today().isoformat(),
        "fuente": args.fuente,
        "recorte": list(recorte) if recorte else None,
        "min_separacion_s": args.min_separacion_s,
        "videos": [],
        "frames": [],
    }

    for ruta in args.video:
        video = Path(ruta)
        print(f"-- {video.name}")
        encontrados, descartes = candidatos(video, args.device, recorte, args.paso)
        print(
            f"   {len(encontrados)} candidatos  "
            f"(sin persona: {descartes['sin_persona']}, sin muñecas: {descartes['sin_munecas']})"
        )
        if not encontrados:
            print("   ningún fotograma utilizable, se salta")
            continue

        if len(encontrados) <= args.tope_por_video:
            elegidos = list(range(len(encontrados)))
        else:
            vectores = [
                pose_vector(np.array(c["keypoints"], dtype=np.float32), np.array(c["box"]))
                for c in encontrados
            ]
            tiempos = [c["t_s"] for c in encontrados]
            elegidos = pick_varied(vectores, tiempos, args.tope_por_video, args.min_separacion_s)
            print(f"   recortados a {len(elegidos)} por el tope, los más variados en postura")

        slug = video.stem.replace(" ", "_")
        manifiesto["videos"].append({"archivo": str(video), "slug": slug, "elegidos": len(elegidos)})
        for i in elegidos:
            c = encontrados[i]
            nombre = f"{slug}_{c['indice']:06d}.jpg"
            cv2.imwrite(str(carpeta_frames / nombre), c["imagen"], [cv2.IMWRITE_JPEG_QUALITY, 95])
            alto, ancho = c["imagen"].shape[:2]
            manifiesto["frames"].append(
                {
                    "archivo": nombre,
                    "video": slug,
                    "fuente": args.fuente,
                    "indice": c["indice"],
                    "t_s": c["t_s"],
                    "ancho": ancho,
                    "alto": alto,
                    "keypoints": c["keypoints"],
                    "scores": c["scores"],
                    "box": c["box"],
                }
            )

    ruta_manifiesto = salida / "manifiesto.json"
    ruta_manifiesto.write_text(json.dumps(manifiesto, indent=2), encoding="utf-8")
    print(f"\n{len(manifiesto['frames'])} fotogramas en {carpeta_frames}")
    print(f"manifiesto: {ruta_manifiesto}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
