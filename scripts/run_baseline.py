"""La línea base: qué saca reconocer la tarea SIN aprender nada.

Corre los cuatro pliegues de validación cruzada por sujeto y compara dos brazos:

  · el DEGENERADO, que contesta siempre la clase más frecuente. Es el suelo: sin
    él, cualquier cifra de acierto es incomparable.
  · las REGLAS geométricas, con los umbrales calibrados en los sujetos de
    ENTRENAMIENTO de cada pliegue. Se les da el mismo privilegio que al modelo para
    que la comparación sea justa: una línea base débil por descuido le regalaría al
    modelo una ventaja que no tiene.

La línea base a batir en cada campo es el MÁXIMO de los dos brazos, no el de las
reglas: en dos de los cuatro campos las reglas pierden contra el degenerado, y
presentarlas como rival en esos casos sería ponerle el listón bajo al modelo.

El reservado no se toca: estos pliegues son de desarrollo. Cuando Juan Diego
entrene su modelo, se mide igual y sobre los mismos pliegues, que es lo único que
hace comparables los números.

    .\\.venv\\Scripts\\python.exe scripts\\run_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline.reba import body_angles  # noqa: E402
from src.baseline.rules import calibrate, predict_all  # noqa: E402
from src.datasets.uwiom import LABEL_FIELDS, load_subject  # noqa: E402
from src.datasets.windows import resample  # noqa: E402
from src.eval.metrics import aggregate, majority_prediction, report  # noqa: E402
from src.eval.splits import cross_validation_folds  # noqa: E402

HZ = 10.0


def subject_frames(index: int):
    """Los fotogramas de un sujeto con persona: esqueleto, flexión del tronco y verdad.

    Los fotogramas sin persona no son culpa del clasificador —no hay postura que
    clasificar— así que se descuentan aquí y no entran en ninguna métrica.
    """
    s = load_subject(index)
    elegidos, _, presente = resample(s, HZ)
    keypoints = s.skeleton.keypoints[elegidos][presente]
    etiquetas = np.array(s.labels)[elegidos][presente]
    return keypoints, body_angles(keypoints)["trunk_flexion"], etiquetas


def main() -> None:
    cache = {}

    def datos(indice):
        if indice not in cache:
            cache[indice] = subject_frames(indice)
        return cache[indice]

    pliegues = cross_validation_folds()
    print(f"Validación cruzada por sujeto, {len(pliegues)} pliegues. El reservado no entra.\n")

    resultados: dict[str, dict[str, list]] = {
        campo: {"reglas": [], "degenerado": []} for campo in LABEL_FIELDS
    }
    resultados["etiqueta completa"] = {"reglas": [], "degenerado": []}

    umbrales_por_pliegue = []
    for pliegue in pliegues:
        entrena = [datos(i) for i in pliegue.train]
        prueba = [datos(i) for i in pliegue.test]

        # Los umbrales salen SOLO de los sujetos de entrenamiento de este pliegue.
        umbrales = calibrate(entrena, HZ)
        umbrales_por_pliegue.append(umbrales)

        predicho = [predict_all(p[0], p[1], HZ, umbrales) for p in prueba]

        for posicion, campo in enumerate(LABEL_FIELDS):
            y_train = np.concatenate([e[2][:, posicion] for e in entrena])
            y_true = np.concatenate([p[2][:, posicion] for p in prueba])
            y_reglas = np.concatenate([pr[campo] for pr in predicho])
            resultados[campo]["reglas"].append(report(y_true, y_reglas))
            resultados[campo]["degenerado"].append(
                report(y_true, majority_prediction(y_train, len(y_true)))
            )

        # La etiqueta entera, que es la tarea tal como la etiqueta el dataset.
        y_train = np.array(["_".join(f) for f in np.concatenate([e[2] for e in entrena])])
        y_true = np.array(["_".join(f) for f in np.concatenate([p[2] for p in prueba])])
        campos_predichos = np.stack(
            [np.concatenate([pr[c] for pr in predicho]) for c in LABEL_FIELDS], axis=1
        )
        y_reglas = np.array(["_".join(f) for f in campos_predichos])
        resultados["etiqueta completa"]["reglas"].append(report(y_true, y_reglas))
        resultados["etiqueta completa"]["degenerado"].append(
            report(y_true, majority_prediction(y_train, len(y_true)))
        )

    print(f"{'campo':<20}{'brazo':<12}{'F1 macro':>10}{'rango':>16}{'accuracy':>10}")
    for campo, brazos in resultados.items():
        for brazo in ("degenerado", "reglas"):
            r = aggregate(brazos[brazo])
            rango = f"{r['macro_f1_min']:.3f}-{r['macro_f1_max']:.3f}"
            print(
                f"{campo:<20}{brazo:<12}{r['macro_f1_median']:>10.3f}"
                f"{rango:>16}{r['accuracy_median']:>10.3f}"
            )
        print()

    print("Umbrales calibrados en cada pliegue (solo con sujetos de entrenamiento):")
    for pliegue, u in zip(pliegues, umbrales_por_pliegue):
        print(
            f"  {pliegue.name}: tronco {u.trunk_bend:.0f} grados, "
            f"andar {u.walk_speed:.2f} cuerpos/s, margen de altura {u.height_margin:.2f}"
        )
    dispersion = np.std([u.trunk_bend for u in umbrales_por_pliegue])
    print(
        f"  desviacion del umbral de tronco entre pliegues: {dispersion:.1f} grados "
        "(si fuera grande, las reglas dependerian de a quien le toco entrenar)\n"
    )

    print("El detalle por clase del campo que decide el informe (movimiento):")
    print(resultados["motion"]["reglas"][0].format("  pliegue 1, reglas calibradas:"))


if __name__ == "__main__":
    main()
