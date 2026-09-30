"""¿Reconoce cajas de almacén un modelo cuando están EN LAS MANOS de alguien?

Es la pregunta que decide si la detección de carga sirve para este sistema, y no se
puede responder con la métrica del entrenamiento. `package-seg` son cajas de
almacén: en estanterías, en cintas, apiladas, enteras y bien iluminadas. Aquí hacen
falta cajas **sostenidas por una persona**, ocluidas por los brazos y el cuerpo,
vistas desde el ángulo de una cámara de planta y a menudo a contraluz.

Un mAP alto en validación no dice nada sobre eso: es el mismo material con el que
se entrenó. La única respuesta honesta sale de mirarlo sobre grabaciones reales.

Se mide sobre tres cosas distintas, de más fácil a más difícil:

  1. Las imágenes de VALIDACIÓN del propio dataset: el suelo, lo que el modelo ya
     sabe hacer. Si aquí falla, el entrenamiento salió mal y no hay más que hablar.
  2. Un vídeo de una persona manipulando una caja: el caso de verdad.
  3. Y dentro de ese vídeo, cuántas de las cajas detectadas se **asocian a las
     manos**, que es lo único que el sistema usa. Detectar una caja del fondo no
     sirve de nada.

    .\\.venv\\Scripts\\python.exe scripts\\probe_load_detector.py --video "C:\\ruta\\video.mp4"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.load.carga import LoadDetector, associate_to_person, body_height_px  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
PESOS_POR_DEFECTO = RAIZ / "artifacts/modelo/carga.pt"


def _pose(frame, modelo_pose, device):
    resultado = modelo_pose.predict(frame, device=device, verbose=False)[0]
    if resultado.boxes is None or not len(resultado.boxes):
        return None
    cajas = resultado.boxes.xyxy.cpu().numpy()
    areas = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
    return resultado.keypoints.data.cpu().numpy()[int(areas.argmax()), :, :2]


def sobre_validacion(detector, carpeta: Path, limite: int = 60) -> None:
    """El suelo: el material con el que se entrenó."""
    imagenes = sorted(list(carpeta.glob("*.jpg")) + list(carpeta.glob("*.png")))[:limite]
    if not imagenes:
        print(f"  (no se encontraron imagenes en {carpeta})")
        return
    con_caja, confianzas = 0, []
    for ruta in imagenes:
        img = cv2.imread(str(ruta))
        if img is None:
            continue
        cajas, confs, _ = detector.detect(img)
        if len(cajas):
            con_caja += 1
            confianzas.append(float(confs.max()))
    print(f"  {con_caja} de {len(imagenes)} imagenes con al menos una caja detectada")
    if confianzas:
        print(f"  confianza de la mejor deteccion: mediana {np.median(confianzas):.2f}")


def sobre_video(detector, modelo_pose, video: Path, device, recorte=None, paso: int = 15) -> None:
    """El caso de verdad: una persona manipulando una caja."""
    captura = cv2.VideoCapture(str(video))
    if not captura.isOpened():
        print(f"  (no se pudo abrir {video})")
        return
    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))

    con_persona = con_caja = asociadas = 0
    confianzas, distancias = [], []
    for f in range(0, total, paso):
        captura.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = captura.read()
        if not ok:
            continue
        if recorte:
            h, w = img.shape[:2]
            img = img[int(recorte[1] * h) : int(recorte[3] * h), int(recorte[0] * w) : int(recorte[2] * w)]

        keypoints = _pose(img, modelo_pose, device)
        if keypoints is None:
            continue
        con_persona += 1

        cajas, confs, _ = detector.detect(img)
        if not len(cajas):
            continue
        con_caja += 1
        confianzas.append(float(confs.max()))

        indice = associate_to_person(cajas, keypoints)
        if indice is not None:
            asociadas += 1
            centro = np.array(
                [
                    (cajas[indice][0] + cajas[indice][2]) / 2,
                    (cajas[indice][1] + cajas[indice][3]) / 2,
                ]
            )
            from src.load.carga import hands_center

            distancias.append(
                float(np.linalg.norm(centro - hands_center(keypoints)) / body_height_px(keypoints))
            )
    captura.release()

    print(f"  {con_persona} fotogramas con persona")
    print(f"  {con_caja} con alguna caja detectada  ({100 * con_caja / max(con_persona, 1):.0f}%)")
    print(
        f"  {asociadas} con la caja asociada a las manos "
        f"({100 * asociadas / max(con_persona, 1):.0f}%)  <- lo unico que el sistema usa"
    )
    if confianzas:
        print(f"  confianza mediana: {np.median(confianzas):.2f}")
    if distancias:
        print(f"  distancia caja-manos: mediana {np.median(distancias):.2f} alturas de cuerpo")


def main() -> None:
    parser = argparse.ArgumentParser(description="¿Sirve el detector de carga aquí?")
    parser.add_argument("--pesos", default=str(PESOS_POR_DEFECTO))
    parser.add_argument("--video", default=None)
    parser.add_argument("--val", default=r"C:\Users\ASUS\Desktop\datasets\package-seg\images\val")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--recorte",
        nargs=4,
        type=float,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="region del video a analizar, en fracciones (para capturas de pantalla)",
    )
    args = parser.parse_args()

    if not Path(args.pesos).exists():
        print(f"No hay pesos en {args.pesos}.")
        print("Cuando termine el entrenamiento, copia best.pt ahi.")
        raise SystemExit(1)

    from ultralytics import YOLO

    device = int(args.device) if args.device.isdigit() else args.device
    detector = LoadDetector(args.pesos, device=device)
    modelo_pose = YOLO("yolo11n-pose.pt")

    print("1) SOBRE LO QUE YA SABE: imagenes de validacion del propio dataset")
    sobre_validacion(detector, Path(args.val))

    if args.video:
        print(f"\n2) SOBRE EL CASO DE VERDAD: {Path(args.video).name}")
        sobre_video(detector, modelo_pose, Path(args.video), device, args.recorte)
        print(
            "\nLECTURA: si en validacion detecta casi siempre y en el video baja mucho, "
            "el modelo no generaliza a cajas en las manos y hay que decirlo en vez de "
            "meterlo igual. Lo que decide es la tercera cifra, no la primera."
        )
        print(
            "\nANTES DE CONCLUIR NADA, MIRAR EL VIDEO. Un 0% tiene DOS causas que estas\n"
            "cifras no separan: que el modelo no generalice, o que en el material no\n"
            "haya ninguna caja de carton que detectar. La primera vez que se corrio\n"
            "esto (30/09) dio 0 de 89 y la causa era la segunda: el objeto sostenido\n"
            "era un organizador de plastico transparente, y no detectarlo era lo\n"
            "correcto. El dataset son cajas de CARTON de almacen, opacas y cerradas.\n"
            "\nPara que esta sonda concluya algo hace falta material con cajas de carton\n"
            "en las manos. Si no lo hay, el resultado es 'no se sabe', no 'no funciona'."
        )


if __name__ == "__main__":
    main()
