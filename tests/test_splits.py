"""Que ningún sujeto cruce la frontera y que el reservado no se pueda tocar sin querer.

    .\\.venv\\Scripts\\python.exe -m tests.test_splits
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.uwiom import SUBJECTS  # noqa: E402
from src.eval.splits import (  # noqa: E402
    DEFAULT_FOLDS,
    DEFAULT_HOLDOUT,
    cross_validation_folds,
    development_subjects,
    holdout,
    holdout_subjects,
)


def test_holdout_and_development_partition_everyone() -> None:
    reservado = set(holdout_subjects())
    desarrollo = set(development_subjects())
    assert not reservado & desarrollo, f"sujetos en los dos lados: {reservado & desarrollo}"
    assert reservado | desarrollo == set(SUBJECTS), "algún sujeto se quedó fuera de la partición"
    assert len(reservado) == DEFAULT_HOLDOUT


def test_holdout_never_appears_in_any_fold() -> None:
    """Es el invariante que da sentido al reservado: si se filtra a un pliegue, ya
    ha participado en decisiones y deja de medir generalización."""
    reservado = set(holdout_subjects())
    for pliegue in cross_validation_folds():
        contaminado = reservado & (set(pliegue.train) | set(pliegue.test))
        assert not contaminado, f"{pliegue.name} contiene reservados: {contaminado}"


def test_every_development_subject_is_tested_exactly_once() -> None:
    """Lo que hace que la validación cruzada valga: todos pasan por prueba, una vez."""
    vistos: list[int] = []
    for pliegue in cross_validation_folds():
        vistos.extend(pliegue.test)
    assert sorted(vistos) == sorted(development_subjects()), (
        f"cobertura mal repartida: {sorted(vistos)}"
    )
    assert len(vistos) == len(set(vistos)), "algún sujeto es prueba en dos pliegues"


def test_train_and_test_never_share_a_subject() -> None:
    for pliegue in cross_validation_folds():
        assert not set(pliegue.train) & set(pliegue.test), pliegue.name
        assert len(pliegue.train) + len(pliegue.test) == len(development_subjects())


def test_partition_is_reproducible() -> None:
    """Dos llamadas dan lo mismo: comparar modelos entrenados con particiones
    distintas no compara los modelos."""
    assert holdout_subjects() == holdout_subjects()
    primero = [(f.name, f.train, f.test) for f in cross_validation_folds()]
    segundo = [(f.name, f.train, f.test) for f in cross_validation_folds()]
    assert primero == segundo


def test_different_seed_gives_a_different_holdout() -> None:
    """Si la semilla no cambiara nada, no habría partición sino una lista fija
    disfrazada, y no se podría comprobar que el resultado no depende de quién cayó
    en el reservado."""
    assert holdout_subjects(seed=1) != holdout_subjects(seed=2)


def test_holdout_is_refused_without_declaring_it() -> None:
    """La barrera del reservado. Sin ella, la regla vive solo en la cabeza de uno."""
    try:
        holdout()
    except PermissionError:
        pass
    else:
        raise AssertionError("devolvió el reservado sin que nadie declarara nada")

    declarado = holdout(declaro_medicion_final=True)
    assert set(declarado.test) == set(holdout_subjects())
    assert not set(declarado.train) & set(declarado.test)


def test_fold_count_is_validated() -> None:
    for k in (0, 1, len(development_subjects()) + 1):
        try:
            cross_validation_folds(k=k)
        except ValueError:
            continue
        raise AssertionError(f"aceptó k={k}")
    assert len(cross_validation_folds()) == DEFAULT_FOLDS


if __name__ == "__main__":
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fallos = 0
    for prueba in pruebas:
        try:
            prueba()
            print(f"  ok   {prueba.__name__}")
        except AssertionError as e:
            fallos += 1
            print(f"  FALLA {prueba.__name__}: {e}")
    print(f"\n{len(pruebas) - fallos}/{len(pruebas)}")
    sys.exit(1 if fallos else 0)
