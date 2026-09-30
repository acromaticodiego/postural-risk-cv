"""El informe que se le entrega a un jefe de planta: riesgo ATRIBUIDO A TAREAS.

Es la pieza que junta las dos mitades del sistema y la única que se puede accionar.
REBA dice cuánto riesgo hay; el modelo dice de qué tarea viene. Por separado no
sirven:

  · «este puesto acumula 40 minutos de riesgo alto» no se puede arreglar;
  · «el 41% de ese riesgo está en colocar cajas a ras del suelo, y lo causa el
    tronco» se arregla subiendo la estantería.

Lo que decide la intervención no es el puntaje: es a qué tarea se le atribuye.

Y de aquí sale la pregunta que de verdad mide si el modelo sirve, que no es su F1:
**¿el informe hecho con las predicciones del modelo lleva a la misma decisión que
el informe hecho con la verdad?** Un modelo que se equivoca a menudo pero acierta
el ranking de tareas ya es suficientemente bueno para este producto, porque el
cliente hace lo mismo en los dos casos. `ranking_agreement` mide exactamente eso.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..baseline.reba import action_level
from .workstation import COMPONENTS, WorkstationConfig


@dataclass(frozen=True)
class TaskExposure:
    """Lo que aporta una tarea al riesgo de un puesto."""

    task: str
    seconds_total: float
    seconds_at_risk: float
    share_of_risk: float
    """Fracción del riesgo TOTAL del puesto que aporta esta tarea. Es la columna
    que ordena la intervención: una tarea peligrosísima que dura diez segundos al
    turno importa menos que una mediana que dura dos horas."""
    median_reba: float
    dominant_component: str


@dataclass(frozen=True)
class TaskReport:
    workstation: str
    tasks: list[TaskExposure]
    total_seconds: float
    total_seconds_at_risk: float
    is_lower_bound: bool

    @property
    def top_task(self) -> TaskExposure | None:
        return self.tasks[0] if self.tasks else None

    def recommendation(self) -> str:
        """La frase que un jefe de planta puede llevar a una reunión."""
        peor = self.top_task
        if peor is None or peor.seconds_at_risk <= 0:
            return "No se detectó exposición por encima del umbral en este puesto."
        traduccion = {
            "trunk": "la flexión del tronco",
            "neck": "la posición del cuello",
            "legs": "la postura de las piernas",
            "upper_arm": "la elevación del brazo",
            "lower_arm": "la posición del antebrazo",
        }
        aviso = (
            " El puesto no está configurado, así que el riesgo real es mayor que el reportado."
            if self.is_lower_bound
            else ""
        )
        return (
            f"El {100 * peor.share_of_risk:.0f}% de la exposición de riesgo de este puesto "
            f"está en «{peor.task}» ({peor.seconds_at_risk / 60:.1f} min). "
            f"La causa dominante es {traduccion.get(peor.dominant_component, peor.dominant_component)}."
            + aviso
        )

    def format(self) -> str:
        lineas = [
            f"PUESTO {self.workstation}  ·  {self.total_seconds / 60:.1f} min observados  ·  "
            f"{self.total_seconds_at_risk / 60:.1f} min en riesgo",
            "",
            f"{'tarea':<32}{'tiempo':>9}{'riesgo':>9}{'% riesgo':>10}{'REBA':>7}  causa",
        ]
        for t in self.tasks:
            lineas.append(
                f"{t.task:<32}{t.seconds_total / 60:>8.1f}m{t.seconds_at_risk / 60:>8.1f}m"
                f"{100 * t.share_of_risk:>9.0f}%{t.median_reba:>7.0f}  {t.dominant_component}"
            )
        lineas += ["", self.recommendation()]
        return "\n".join(lineas)


def build_report(
    tasks: np.ndarray,
    scores: dict[str, np.ndarray],
    hz: float,
    config: WorkstationConfig,
    present: np.ndarray | None = None,
    min_seconds: float = 5.0,
) -> TaskReport:
    """Cruza la tarea de cada fotograma con su puntaje de riesgo.

    `min_seconds` deja fuera del informe las tareas que apenas aparecen. No es para
    que quede bonito: una tarea de tres segundos da un porcentaje inestable que
    puede encabezar el ranking por azar, y el ranking es lo que dispara una
    inversión. El tiempo descartado sigue contando en el total.
    """
    reba = np.asarray(scores["reba"], dtype=int)
    if present is None:
        present = reba > 0
    dt = 1.0 / hz
    en_riesgo = present & (reba >= config.risk_threshold)
    riesgo_total = float(en_riesgo.sum()) * dt

    filas: list[TaskExposure] = []
    for tarea in np.unique(tasks[present]):
        de_esta = present & (tasks == tarea)
        segundos = float(de_esta.sum()) * dt
        if segundos < min_seconds:
            continue
        riesgo_aqui = de_esta & en_riesgo
        medias = {c: float(np.asarray(scores[c])[riesgo_aqui].mean()) for c in COMPONENTS} if riesgo_aqui.any() else {c: 0.0 for c in COMPONENTS}
        filas.append(
            TaskExposure(
                task=str(tarea),
                seconds_total=segundos,
                seconds_at_risk=float(riesgo_aqui.sum()) * dt,
                share_of_risk=float(riesgo_aqui.sum()) * dt / riesgo_total if riesgo_total else 0.0,
                median_reba=float(np.median(reba[de_esta])),
                dominant_component=max(medias, key=medias.get),
            )
        )

    filas.sort(key=lambda t: -t.seconds_at_risk)
    return TaskReport(
        workstation=config.id,
        tasks=filas,
        total_seconds=float(present.sum()) * dt,
        total_seconds_at_risk=riesgo_total,
        is_lower_bound=not config.is_configured,
    )


def ranking_agreement(truth: TaskReport, predicted: TaskReport, top_n: int = 3) -> dict:
    """¿Lleva el informe del modelo a la misma decisión que el de la verdad?

    Es la medida que importa para el producto. Un modelo con un F1 mediocre que
    señala las mismas tareas prioritarias ya sirve, porque el cliente interviene
    igual; uno con buen F1 que cambia el orden de las prioridades, no.

    Se miran tres cosas, de más a menos exigente: si la tarea número uno coincide,
    cuántas de las `top_n` coinciden como conjunto, y cuánto se desvía la fracción
    de riesgo atribuida a la peor tarea.
    """
    t_top = [t.task for t in truth.tasks[:top_n]]
    p_top = [t.task for t in predicted.tasks[:top_n]]
    peor_verdad = truth.top_task
    peor_predicho = predicted.top_task
    return {
        "top1_agree": bool(t_top and p_top and t_top[0] == p_top[0]),
        "topn_overlap": len(set(t_top) & set(p_top)),
        "topn": top_n,
        "share_truth": peor_verdad.share_of_risk if peor_verdad else 0.0,
        "share_predicted": peor_predicho.share_of_risk if peor_predicho else 0.0,
        "risk_minutes_truth": truth.total_seconds_at_risk / 60,
        "risk_minutes_predicted": predicted.total_seconds_at_risk / 60,
    }
