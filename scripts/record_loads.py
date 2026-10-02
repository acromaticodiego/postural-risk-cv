"""Grabar la cámara EN CRUDO, para tener con qué afinar el detector de carga.

POR QUÉ HACE FALTA UN GUION PARA ESTO. Los cuatro vídeos que había eran capturas de
pantalla del panel, y una captura del panel no sirve para entrenar por dos razones
que se ven de golpe al mirarla: el esqueleto va PINTADO sobre la persona y sobre la
carga —así que el modelo aprendería a buscar líneas de colores—, y la imagen de la
cámara ocupa un recuadro del navegador que luego se vuelve a comprimir. Medido sobre
una de ellas: YOLO detecta a la persona con confianza 0,13 donde el sistema en vivo
la detectaba sin problemas, y en 148 de 317 fotogramas no la encuentra en absoluto.

Esto abre la cámara y escribe lo que ve, sin pasar por el navegador y sin dibujar
nada encima. La ventana de previsualización es la cámara tal cual, a propósito: lo
que se ve es lo que se graba.

QUÉ GRABAR, que importa más que el guion:

  · **La carga de verdad**, la que el sistema va a tener que reconocer. Si en la
    planta son cajas de cartón, cajas de cartón.
  · **Dos o tres objetos distintos**, y cada uno en su propio vídeo. La partición de
    `finetune_load.py` es POR VÍDEO, así que un objeto que solo aparece en un vídeo
    permite preguntar si el modelo generaliza al siguiente.
  · **Levantando, cargando y soltando**, y desde el suelo y desde una mesa. Los
    fotogramas útiles son los que tienen la carga en las manos; de pie sin nada no
    sirven de nada para esto.
  · **De lado**, que es además la única vista desde la que REBA se puede medir —el
    sistema ya avisa cuando la cámara está de frente—. Así el mismo material vale
    para las dos cosas.
  · **Dos o tres minutos por vídeo basta**: de 50 segundos salieron 138 fotogramas
    candidatos.

    .\\.venv\\Scripts\\python.exe scripts\\record_loads.py --nombre carga_lateral
    # q o Esc para parar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "data/carga/videos"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nombre", required=True, help="un nombre por objeto, sin espacios")
    parser.add_argument("--camara", type=int, default=0)
    parser.add_argument("--ancho", type=int, default=1280)
    parser.add_argument("--alto", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--destino", default=str(DESTINO))
    args = parser.parse_args()

    destino = Path(args.destino)
    destino.mkdir(parents=True, exist_ok=True)
    salida = destino / f"{args.nombre}.mp4"
    if salida.exists():
        print(f"ya existe {salida}. Elige otro nombre o bórralo.")
        return 2

    captura = cv2.VideoCapture(args.camara, cv2.CAP_DSHOW)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, args.ancho)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, args.alto)
    if not captura.isOpened():
        print(f"no se pudo abrir la cámara {args.camara}")
        return 2

    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (ancho, alto) != (args.ancho, args.alto):
        print(f"la cámara dio {ancho}x{alto} en vez de {args.ancho}x{args.alto}")

    escritor = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (ancho, alto))
    print(f"grabando {ancho}x{alto} en {salida}")
    print("q o Esc para parar\n")

    n = 0
    try:
        while True:
            ok, fotograma = captura.read()
            if not ok:
                break
            escritor.write(fotograma)
            n += 1

            # La previsualización se dibuja sobre una COPIA. El contador no puede
            # acabar grabado en el vídeo: sería exactamente el fallo de las capturas
            # de pantalla, píxeles de interfaz dentro del material de entrenamiento.
            vista = fotograma.copy()
            cv2.putText(
                vista,
                f"{n / args.fps:5.1f} s   [q] parar",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (30, 230, 120),
                2,
            )
            cv2.imshow("grabando la camara en crudo", vista)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        captura.release()
        escritor.release()
        cv2.destroyAllWindows()

    print(f"{n} fotogramas, {n / args.fps:.1f} s")
    print(f"\nahora:\n  .\\.venv\\Scripts\\python.exe scripts\\extract_load_frames.py "
          f'--video "{salida}" --fuente camara')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
