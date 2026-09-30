"""¿Cómo distingue el modelo una caja de una varilla, si eso no está en el esqueleto?

El TCN saca 0,763 de F1 macro en el campo `object` (caja / varilla / nada), y la
guía del proyecto tenía escrito de antemano que ese resultado había que mirarlo con
lupa: **el objeto no es visible en un esqueleto**, así que acertarlo tiene dos
explicaciones muy distintas.

  H1. LEGÍTIMA: manipular una caja y manipular una varilla son posturas distintas.
      Una caja se coge con las dos manos separadas por su ancho; una varilla se
      coge con las manos juntas o con una sola. Eso SÍ está en el esqueleto, y el
      modelo no estaría viendo el objeto sino la forma de agarrarlo — que para un
      sistema de ergonomía es exactamente lo que interesa.
  H2. TRAMPA: los 20 participantes ejecutan el guion en el mismo orden, así que
      podría existir alguna pista de contexto que permita acertar sin mirar la
      postura.

La prueba: si H1 es cierta, la distancia entre las muñecas tiene que separar `box`
de `rod` DE FORMA VISIBLE en los datos crudos, sin ningún modelo de por medio. Si
no las separa, el modelo está sacando el acierto de otro sitio y hay que buscarlo.

    .\\.venv\\Scripts\\python.exe scripts\\probe_object_leak.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.harness import load  # noqa: E402
from src.eval.splits import development_subjects  # noqa: E402
from src.pose.schema import JOINT  # noqa: E402


def main() -> None:
    por_objeto: dict[str, list[float]] = {}
    manos_altura: dict[str, list[float]] = {}

    for i in development_subjects():
        s = load(i)
        kp = s.keypoints
        alto = np.maximum(kp[:, :, 1].max(axis=1) - kp[:, :, 1].min(axis=1), 1.0)
        separacion = (
            np.linalg.norm(
                kp[:, JOINT["left_wrist"], :] - kp[:, JOINT["right_wrist"], :], axis=1
            )
            / alto
        )
        # Diferencia de altura entre las dos manos: sostener una varilla larga suele
        # dejar una mano más alta que la otra; una caja se lleva a la misma altura.
        desnivel = (
            np.abs(kp[:, JOINT["left_wrist"], 1] - kp[:, JOINT["right_wrist"], 1]) / alto
        )
        for objeto in np.unique(s.labels[:, 0]):
            mascara = s.labels[:, 0] == objeto
            por_objeto.setdefault(objeto, []).extend(separacion[mascara].tolist())
            manos_altura.setdefault(objeto, []).extend(desnivel[mascara].tolist())

    print("Separacion entre munecas, en alturas de cuerpo (solo sujetos de desarrollo)")
    print(f"{'objeto':<10}{'n':>9}{'p25':>8}{'mediana':>9}{'p75':>8}")
    for objeto, valores in sorted(por_objeto.items()):
        v = np.array(valores)
        print(
            f"{objeto:<10}{len(v):>9}{np.percentile(v, 25):>8.3f}"
            f"{np.median(v):>9.3f}{np.percentile(v, 75):>8.3f}"
        )

    print("\nDesnivel entre las dos manos, en alturas de cuerpo")
    print(f"{'objeto':<10}{'n':>9}{'p25':>8}{'mediana':>9}{'p75':>8}")
    for objeto, valores in sorted(manos_altura.items()):
        v = np.array(valores)
        print(
            f"{objeto:<10}{len(v):>9}{np.percentile(v, 25):>8.3f}"
            f"{np.median(v):>9.3f}{np.percentile(v, 75):>8.3f}"
        )

    caja = np.array(por_objeto.get("box", []))
    varilla = np.array(por_objeto.get("rod", []))
    if len(caja) and len(varilla):
        # Solapamiento de los rangos intercuartílicos: si las dos cajas de bigotes
        # se montan del todo, la separación de muñecas no distingue nada.
        c25, c75 = np.percentile(caja, [25, 75])
        v25, v75 = np.percentile(varilla, [25, 75])
        solapa = max(0.0, min(c75, v75) - max(c25, v25))
        union = max(c75, v75) - min(c25, v25)
        print(
            f"\nCaja mediana {np.median(caja):.3f} contra varilla {np.median(varilla):.3f}; "
            f"solapamiento intercuartilico {100 * solapa / union:.0f}%"
        )
        print(
            "VEREDICTO: si las medianas se separan y el solapamiento no es total, H1 "
            "se sostiene: el modelo lee la FORMA DE AGARRAR, no el objeto."
        )


if __name__ == "__main__":
    main()
