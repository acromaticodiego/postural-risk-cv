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
    parser.add_argument(
        "--cuenta-atras",
        type=float,
        default=5.0,
        help="segundos para colocarse antes de que empiece a grabar. 0 para empezar ya",
    )
    parser.add_argument(
        "--rehacer",
        action="store_true",
        help="sobrescribe una toma anterior con el mismo nombre",
    )
    args = parser.parse_args()

    destino = Path(args.destino)
    destino.mkdir(parents=True, exist_ok=True)
    salida = destino / f"{args.nombre}.mp4"
    if salida.exists() and not args.rehacer:
        # Negarse por defecto es lo correcto —una toma buena no se pisa sin querer—
        # pero la primera toma de alguien dura dos segundos y la va a repetir, así que
        # la salida tiene que estar aquí y no en pedirle a otro que borre el fichero.
        print(f"ya existe {salida}")
        print(f"para repetir la toma:  ...record_loads.py --nombre {args.nombre} --rehacer")
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

    # Antes de grabar, tiempo para apartarse del teclado y colocarse con la carga. Sin
    # esto la toma empieza con alguien sentado delante del portátil, que no es una
    # postura de trabajo y encima son los primeros fotogramas, los que más se miran.
    if args.cuenta_atras > 0:
        import time

        arranque = time.monotonic()
        while True:
            restante = args.cuenta_atras - (time.monotonic() - arranque)
            if restante <= 0:
                break
            ok, fotograma = captura.read()
            if not ok:
                break
            vista = fotograma.copy()
            cv2.putText(
                vista,
                f"{restante:.0f}",
                (ancho // 2 - 40, alto // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                4.0,
                (40, 200, 255),
                6,
            )
            cv2.putText(
                vista,
                "colocate con la carga",
                (ancho // 2 - 190, alto // 2 + 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (40, 200, 255),
                2,
            )
            cv2.imshow("grabando la camara en crudo", vista)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                captura.release()
                cv2.destroyAllWindows()
                print("cancelado antes de empezar")
                return 1

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

    segundos = n / args.fps
    print(f"{n} fotogramas, {segundos:.1f} s")
    if segundos < 60:
        print(f"\nESO ES POCO. Con {segundos:.0f} s no salen fotogramas variados suficientes;")
        print("la referencia son 2 o 3 minutos por objeto. Para repetir la toma:")
        print(f"  .\\.venv\\Scripts\\python.exe scripts\\record_loads.py "
              f"--nombre {args.nombre} --rehacer")
        return 0

    print(f"\nahora:\n  .\\.venv\\Scripts\\python.exe scripts\\extract_load_frames.py "
          f'--video "{salida}" --fuente camara')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
