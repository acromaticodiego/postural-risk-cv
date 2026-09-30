"""Quién entrena y quién evalúa. La partición es POR SUJETO, siempre.

Si una persona aparece en entrenamiento y en prueba, el modelo puede memorizar su
cuerpo y su forma de moverse, y el número sale inflado sin que nada avise. Es el
fallo más común en reconocimiento de acción sobre esqueletos, y este módulo existe
para que no pueda ocurrir por descuido: las particiones se construyen sobre la
lista de sujetos y nunca sobre la de fotogramas ni de ventanas.

DOS MECANISMOS, Y POR QUÉ LOS DOS

  · VALIDACIÓN CRUZADA POR SUJETO, para desarrollar. Con 20 participantes, una
    partición fija de cuatro para prueba es frágil: un sujeto vale el 25% del
    resultado, así que cambiar de persona cambia la conclusión. Rotando los
    pliegues, todos los sujetos son prueba una vez y el número sale con su
    dispersión en vez de con una cifra suelta.
  · RESERVADO que no entra en ningún pliegue, para la medición final. Existe
    porque la validación cruzada se usa para tomar decisiones —qué arquitectura,
    qué ventana, qué umbral— y un conjunto con el que se han tomado decisiones ya
    no mide generalización, mide ajuste.

UN LÍMITE QUE HAY QUE DECIR: la ficha de UW-IOM dice que hay 15 hombres y 5
mujeres, pero NO dice quién es quién. Así que la partición no puede estratificar
por sexo, y el reservado podría quedarse sin ninguna mujer por azar. No se intenta
adivinar a partir del cuerpo. Lo que sí mitiga el problema es la validación
cruzada, donde todos los sujetos pasan por prueba.

EL RESERVADO SE MIRA UNA VEZ. Un reservado usado no vuelve a ser un reservado, así
que `holdout()` exige declararlo a propósito.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ..datasets.uwiom import SUBJECTS

RAIZ = Path(__file__).resolve().parents[2]
REGISTRO = RAIZ / "artifacts/particion.json"

# PROPUESTA, pendiente de que Juan Diego la confirme: 4 sujetos reservados y 4
# pliegues de 4 sobre los 16 restantes. Con 20 sujetos es el reparto que deja el
# reservado lo bastante grande para no depender de una persona y los pliegues lo
# bastante grandes para entrenar.
DEFAULT_HOLDOUT = 4
DEFAULT_FOLDS = 4

# La semilla se fija y se escribe en el registro. Una partición que dependa de la
# hora a la que se corrió no es reproducible, y comparar dos modelos entrenados
# con particiones distintas no compara los modelos.
SEED = 20260929


@dataclass(frozen=True)
class Split:
    """Un pliegue: qué sujetos entrenan y qué sujetos evalúan."""

    name: str
    train: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        solapan = set(self.train) & set(self.test)
        if solapan:
            raise ValueError(f"{self.name}: los sujetos {solapan} están en los dos lados")


def _shuffled_subjects(seed: int = SEED) -> list[int]:
    """Orden determinista de los sujetos, sin depender de numpy ni del entorno."""
    import random

    orden = list(SUBJECTS)
    random.Random(seed).shuffle(orden)
    return orden


def holdout_subjects(n: int = DEFAULT_HOLDOUT, seed: int = SEED) -> tuple[int, ...]:
    """Los sujetos reservados. Saber QUIÉNES son no es mirarlos."""
    return tuple(sorted(_shuffled_subjects(seed)[:n]))


def development_subjects(n_holdout: int = DEFAULT_HOLDOUT, seed: int = SEED) -> tuple[int, ...]:
    return tuple(sorted(_shuffled_subjects(seed)[n_holdout:]))


def cross_validation_folds(
    k: int = DEFAULT_FOLDS, n_holdout: int = DEFAULT_HOLDOUT, seed: int = SEED
) -> list[Split]:
    """k pliegues sobre los sujetos de desarrollo. El reservado no entra en ninguno."""
    desarrollo = _shuffled_subjects(seed)[n_holdout:]
    if k < 2 or k > len(desarrollo):
        raise ValueError(f"k={k} no cabe con {len(desarrollo)} sujetos de desarrollo")

    bloques: list[list[int]] = [[] for _ in range(k)]
    for posicion, sujeto in enumerate(desarrollo):
        bloques[posicion % k].append(sujeto)

    pliegues = []
    for indice, prueba in enumerate(bloques):
        entrena = [s for s in desarrollo if s not in prueba]
        pliegues.append(
            Split(
                name=f"fold{indice + 1}",
                train=tuple(sorted(entrena)),
                test=tuple(sorted(prueba)),
            )
        )
    return pliegues


def holdout(declaro_medicion_final: bool = False, n: int = DEFAULT_HOLDOUT) -> Split:
    """El reservado, y solo si se declara que esto es la medición final.

    La barrera es deliberadamente incómoda. En el proyecto del agente de voz, el
    mismo mecanismo es lo que impidió gastar un conjunto reservado a medias, y la
    lección fue que la protección tiene que estar en el código: recordar la regla
    no basta cuando quedan cinco minutos y apetece ver el número.
    """
    if not declaro_medicion_final:
        raise PermissionError(
            "El reservado se mide UNA vez. Si de verdad es la medición final, "
            "llama a holdout(declaro_medicion_final=True) y apunta la fecha en "
            "artifacts/particion.json. Para desarrollar, usa cross_validation_folds()."
        )
    return Split(
        name="holdout",
        train=development_subjects(n),
        test=holdout_subjects(n),
    )


def write_registry(k: int = DEFAULT_FOLDS, n_holdout: int = DEFAULT_HOLDOUT) -> dict:
    """Deja la partición escrita, para que se pueda auditar y reproducir."""
    registro = {
        "created_on": date.today().isoformat(),
        "seed": SEED,
        "n_subjects": len(SUBJECTS),
        "holdout_subjects": list(holdout_subjects(n_holdout)),
        "development_subjects": list(development_subjects(n_holdout)),
        "folds": [
            {"name": f.name, "train": list(f.train), "test": list(f.test)}
            for f in cross_validation_folds(k, n_holdout)
        ],
        "holdout_measured_on": None,
        "note": (
            "Partición POR SUJETO. El reservado no entra en ningún pliegue y se "
            "mide una sola vez; cuando se mida, anotar la fecha en "
            "holdout_measured_on y no volver a tocarlo."
        ),
    }
    REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    REGISTRO.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    return registro


if __name__ == "__main__":
    registro = write_registry()
    print(f"semilla {registro['seed']}, {registro['n_subjects']} sujetos")
    print(f"reservado: {registro['holdout_subjects']}  (no entra en ningún pliegue)")
    for pliegue in registro["folds"]:
        print(f"  {pliegue['name']}: prueba {pliegue['test']}")
    print(f"\nescrito en {REGISTRO.relative_to(RAIZ)}")
