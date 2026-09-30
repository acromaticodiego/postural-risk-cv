"""Cómo se puntúa un clasificador aquí, y por qué no con `accuracy`.

Las clases de UW-IOM están brutalmente desbalanceadas: `stand / place` tiene 12.010
fotogramas y `walk / hold` 388. Un clasificador que conteste siempre la clase más
frecuente acierta una fracción respetable y **no ha aprendido nada**, así que la
cifra principal es el F1 MACRO —la media de los F1 por clase, donde una clase rara
pesa lo mismo que una común— y `accuracy` se publica al lado solo para que se vea
la diferencia.

Y siempre acompañado de la LÍNEA BASE DEGENERADA: qué saca el que contesta siempre
lo mismo. Sin ese número, ninguna cifra de acierto significa nada.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassScore:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class Report:
    classes: list[ClassScore]
    accuracy: float
    macro_f1: float
    n: int

    def format(self, title: str = "") -> str:
        lineas = []
        if title:
            lineas.append(title)
        lineas.append(f"{'clase':<22}{'prec':>7}{'rec':>7}{'F1':>7}{'n':>8}")
        for c in sorted(self.classes, key=lambda c: -c.support):
            lineas.append(
                f"{c.label:<22}{c.precision:>7.2f}{c.recall:>7.2f}{c.f1:>7.2f}{c.support:>8}"
            )
        lineas.append(
            f"{'':<22}{'':>7}{'':>7}{self.macro_f1:>7.2f}{self.n:>8}   <- F1 macro"
        )
        lineas.append(f"accuracy {self.accuracy:.3f}  (engaña: ver la línea base degenerada)")
        return "\n".join(lineas)


def report(y_true: np.ndarray, y_pred: np.ndarray) -> Report:
    """F1 por clase sobre la unión de clases vistas y predichas.

    Se toman las de los dos lados a propósito: un clasificador que invente una clase
    que no existe en la verdad tiene que verse penalizado, y si solo se miraran las
    clases verdaderas, sus falsos positivos desaparecerían del informe.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    etiquetas = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    puntajes = []
    for etiqueta in etiquetas:
        verdadero = y_true == etiqueta
        predicho = y_pred == etiqueta
        tp = int((verdadero & predicho).sum())
        fp = int((~verdadero & predicho).sum())
        fn = int((verdadero & ~predicho).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        puntajes.append(ClassScore(etiqueta, precision, recall, f1, int(verdadero.sum())))

    # El F1 macro promedia solo sobre clases que EXISTEN en la verdad: una clase
    # inventada por el clasificador ya está penalizada en la precisión de esa clase,
    # y meterla en el promedio con soporte 0 castigaría dos veces lo mismo.
    presentes = [p for p in puntajes if p.support > 0]
    return Report(
        classes=puntajes,
        accuracy=float((y_true == y_pred).mean()),
        macro_f1=float(np.mean([p.f1 for p in presentes])) if presentes else 0.0,
        n=len(y_true),
    )


def majority_prediction(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """El clasificador degenerado: siempre la clase más frecuente del entrenamiento.

    La clase mayoritaria se saca de TRAIN y no de test. Sacarla de test sería mirar
    las respuestas, y este número existe precisamente para ser un suelo honesto.
    """
    valores, cuentas = np.unique(np.asarray(y_train), return_counts=True)
    return np.full(n_test, valores[cuentas.argmax()])


def aggregate(reports: list[Report]) -> dict[str, float]:
    """Resumen de varios pliegues: mediana y rango, nunca una cifra sola.

    Con cuatro sujetos por pliegue, la diferencia entre pliegues es información
    —dice cuánto depende el resultado de a quién le tocó evaluar— y promediarla en
    un número la esconde.
    """
    macro = np.array([r.macro_f1 for r in reports])
    acc = np.array([r.accuracy for r in reports])
    return {
        "macro_f1_median": float(np.median(macro)),
        "macro_f1_min": float(macro.min()),
        "macro_f1_max": float(macro.max()),
        "accuracy_median": float(np.median(acc)),
        "folds": len(reports),
    }
