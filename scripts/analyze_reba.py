"""El primer número real: REBA geométrico sobre los 20 sujetos de UW-IOM.

Y sobre todo, LA COMPROBACIÓN QUE LE DA CREDIBILIDAD AL RESTO: el dataset no trae
puntajes de riesgo, así que no hay verdad de terreno contra la que medir el REBA.
Lo que sí trae son etiquetas de acción, y de ahí sale una validación indirecta pero
fuerte: **si el REBA calculado no puntúa más alto agachándose a recoger del suelo
que estando de pie, el cálculo no vale**, por muy bien que estén copiadas las
tablas.

    .\\.venv\\Scripts\\python.exe scripts\\analyze_reba.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline.reba import reba_from_keypoints  # noqa: E402
from src.datasets.uwiom import SUBJECTS, load_subject  # noqa: E402
from src.datasets.windows import resample  # noqa: E402
from src.product.workstation import WorkstationConfig, summarize_exposure  # noqa: E402

HZ = 10.0
CONFIG = WorkstationConfig(id="UWIOM", name="Laboratorio UW-IOM")


def main() -> None:
    por_sujeto = []
    reba_por_accion: dict[tuple[str, str], list[int]] = defaultdict(list)
    reba_por_altura: dict[str, list[int]] = defaultdict(list)
    total_eventos = 0
    causas: dict[str, int] = defaultdict(int)

    for i in SUBJECTS:
        s = load_subject(i)
        elegidos, _, presente = resample(s, HZ)
        keypoints = s.skeleton.keypoints[elegidos]
        etiquetas = np.array(s.labels)[elegidos]

        datos = reba_from_keypoints(keypoints, present=presente, assumptions=CONFIG.assumptions())
        resumen = summarize_exposure(datos, HZ, CONFIG, present=presente)
        reba = datos["reba"]

        por_sujeto.append(
            (
                i,
                resumen.measured_seconds,
                resumen.seconds_at_risk,
                100 * resumen.seconds_at_risk / max(resumen.measured_seconds, 1e-9),
                len(resumen.events),
                float(np.median(reba[presente])),
                s.fps,
            )
        )
        total_eventos += len(resumen.events)
        for e in resumen.events:
            causas[e.dominant_component] += 1

        for fotograma in range(len(reba)):
            if not presente[fotograma]:
                continue
            movimiento, manipulacion = etiquetas[fotograma][1], etiquetas[fotograma][2]
            reba_por_accion[(movimiento, manipulacion)].append(int(reba[fotograma]))
            reba_por_altura[etiquetas[fotograma][3]].append(int(reba[fotograma]))

    print("REBA GEOMETRICO SOBRE UW-IOM, 20 sujetos, 10 Hz, puesto sin configurar")
    print("(sin configurar = asunciones neutras = COTA INFERIOR del riesgo real)\n")
    print("suj  medido(s)  en riesgo(s)    %   eventos  REBA mediana  fps real")
    for fila in por_sujeto:
        print(
            "%3d %10.0f %13.1f %5.1f %9d %13.0f %9.2f"
            % (fila[0], fila[1], fila[2], fila[3], fila[4], fila[5], fila[6])
        )

    pct = np.array([f[3] for f in por_sujeto])
    print(
        f"\n20 sujetos | tiempo en riesgo: mediana {np.median(pct):.1f}%, "
        f"rango {pct.min():.1f}-{pct.max():.1f}% | {total_eventos} eventos en total"
    )
    print("causa dominante de los eventos:", dict(sorted(causas.items(), key=lambda kv: -kv[1])))

    print("\n--- LA COMPROBACION: ¿distingue el REBA lo que dicen las etiquetas? ---")
    print("\nmovimiento + manipulacion        n      REBA mediana   % >= 4")
    filas = sorted(reba_por_accion.items(), key=lambda kv: -float(np.median(kv[1])))
    for (movimiento, manipulacion), valores in filas:
        if len(valores) < 200:
            continue
        v = np.array(valores)
        print(
            "%-30s %7d %11.0f %10.1f"
            % (f"{movimiento} / {manipulacion}", len(v), np.median(v), 100 * (v >= 4).mean())
        )

    print("\naltura de la superficie          n      REBA mediana   % >= 4")
    for altura, valores in sorted(reba_por_altura.items(), key=lambda kv: -float(np.median(kv[1]))):
        v = np.array(valores)
        print("%-30s %7d %11.0f %10.1f" % (altura, len(v), np.median(v), 100 * (v >= 4).mean()))

    doblado = np.array([r for (m, _), vs in reba_por_accion.items() if m == "bend" for r in vs])
    de_pie = np.array([r for (m, _), vs in reba_por_accion.items() if m == "stand" for r in vs])
    print(
        f"\nVEREDICTO: agachado mediana {np.median(doblado):.0f} (n={len(doblado)}), "
        f"de pie mediana {np.median(de_pie):.0f} (n={len(de_pie)}); "
        f"en riesgo el {100 * (doblado >= 4).mean():.0f}% contra el {100 * (de_pie >= 4).mean():.0f}%"
    )


if __name__ == "__main__":
    main()
