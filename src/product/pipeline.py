"""De esqueletos a lo que ve un jefe de planta: puestos, turnos, eventos y replays.

Es el pipeline que un despliegue real correría cada noche sobre las grabaciones del
día. Aquí se ejecuta sobre UW-IOM tratando a cada participante como un turno y
agrupándolos en puestos de trabajo, que es la forma más honesta de enseñar lo que
haría el sistema instalado: un puesto con varios turnos acumulados.

UNA REGLA QUE SE CUMPLE Y HAY QUE DECIR: cada turno se predice con el modelo del
pliegue que NO lo vio entrenando. La demo enseña predicciones genuinas sobre gente
que el modelo no conoce, que es lo mismo que le pasaría en una planta. Enseñar
predicciones sobre los datos de entrenamiento daría una demo más lucida y sería
mentir.

Lo que se guarda para el navegador son ESQUELETOS, nunca imágenes. El replay de un
evento son unos cientos de números.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..baseline.niosh import analyze_lift
from ..baseline.reba import action_level, reba_from_keypoints
from ..eval.harness import HZ
from .report import build_report
from .workstation import WorkstationConfig, summarize_exposure

# Los puestos de la demostración. Cada uno agrupa los turnos de un pliegue, con una
# configuración distinta para que el ranking tenga algo que ordenar: en una planta
# real dos puestos no manejan el mismo peso ni el mismo tipo de agarre.
MAX_REPLAYS = 12

DEMO_WORKSTATIONS = (
    WorkstationConfig(
        id="LINEA-3",
        name="Linea 3 - surtido de cajas",
        load_kg=12.0,
        coupling="fair",
        worker_height_cm=172.0,
        lifts_per_min=4.0,
    ),
    # 18 kg con agarre malo, pero SIN carga brusca. Con `sudden_load=True` este
    # puesto salía con el 100% del tiempo en riesgo, y no por un fallo: medido, una
    # persona de pie y erguida en esa configuración ya puntúa REBA 4, que es el
    # umbral de acción. La norma lo dice así y tiene razón, pero un puesto que está
    # siempre en rojo no enseña nada sobre la postura. Ver la nota de la GUIA sobre
    # el suelo de riesgo del puesto.
    WorkstationConfig(
        id="ALMACEN-1",
        name="Almacen 1 - estanteria baja",
        load_kg=18.0,
        coupling="poor",
        worker_height_cm=168.0,
        lifts_per_min=6.0,
    ),
    WorkstationConfig(
        id="EMPAQUE-2",
        name="Empaque 2 - mesa de armado",
        load_kg=4.0,
        coupling="good",
        worker_height_cm=165.0,
        lifts_per_min=2.0,
    ),
    # A propósito sin configurar: enseña en pantalla qué pasa cuando el cliente aún
    # no ha declarado el peso, y que entonces el informe se marca como cota inferior.
    WorkstationConfig(id="RECIBO-1", name="Recibo 1 - sin configurar"),
)


@dataclass(frozen=True)
class EventReplay:
    """Un evento con lo justo para dibujarlo en el navegador."""

    id: str
    workstation: str
    shift: str
    start_seconds: float
    duration_seconds: float
    peak_reba: int
    peak_level: str
    dominant_component: str
    task: str
    keypoints: list
    """(T, 17, 2) ya escalado a un lienzo de 0..1, para que el navegador no tenga
    que saber nada de la resolución de la cámara original."""
    reba: list
    components: list
    niosh: dict | None = None
    """El análisis NIOSH, solo si el evento es un LEVANTAMIENTO. La ecuación mide
    levantar una carga, no estar de pie en mala postura: aplicarla a un evento de
    alcance o de sostener daría un índice sin significado."""


# Un evento es un levantamiento si su tarea dominante lo dice. La ecuación NIOSH no
# se aplica a cualquier postura de riesgo.
LIFTING_TASKS = ("pick-up", "place")


def _is_lift(task: str) -> bool:
    return any(t in task for t in LIFTING_TASKS)


def _normalize_for_canvas(keypoints: np.ndarray) -> np.ndarray:
    """Encaja el esqueleto en un cuadrado 0..1, con UNA transformación para todo el
    tramo: reencuadrar fotograma a fotograma haría que la persona pareciera quieta."""
    puntos = keypoints.reshape(-1, 2)
    puntos = puntos[(puntos != 0).any(axis=1)]
    if not len(puntos):
        return np.zeros_like(keypoints)
    minimo, maximo = puntos.min(axis=0), puntos.max(axis=0)
    escala = 1.0 / max(float((maximo - minimo).max()), 1.0)
    centro = (minimo + maximo) / 2
    return (keypoints - centro) * escala + 0.5


def process_shift(
    subject_data,
    task_predictions: dict[str, np.ndarray],
    config: WorkstationConfig,
) -> dict:
    """Un turno: puntúa, resume y extrae los replays de sus eventos."""
    keypoints = subject_data.keypoints[subject_data.evaluable]
    scores = reba_from_keypoints(keypoints, assumptions=config.assumptions())
    tareas = np.array(
        [
            f"{m} / {mn} / {h}"
            for m, mn, h in zip(
                task_predictions["motion"],
                task_predictions["manipulation"],
                task_predictions["height"],
            )
        ]
    )

    resumen = summarize_exposure(scores, HZ, config)
    informe = build_report(tareas, scores, HZ, config)

    # La frecuencia se DECLARA, no se mide, y esto costó un número inflado antes de
    # entenderlo. NIOSH supone que la frecuencia se sostiene durante horas; contarla
    # sobre tres minutos de grabación en los que alguien levanta sin parar da una
    # frecuencia de experimento que hunde el peso recomendado, y entonces el factor
    # que más penaliza sale siempre «frecuencia» y tapa lo que de verdad pasa. La
    # medida se conserva al lado para poder compararla con lo declarado.
    levantamientos = sum(
        1
        for e in resumen.events
        if _is_lift(str(np.unique(tareas[e.start_frame : e.end_frame + 1])[0]))
    )
    medida = max(levantamientos / max(resumen.measured_seconds / 60, 1e-9), 0.2)
    por_minuto = config.lifts_per_min if config.lifts_per_min else medida
    frecuencia_estimada = config.lifts_per_min is None

    replays = []
    for n, evento in enumerate(resumen.events):
        corte = slice(evento.start_frame, evento.end_frame + 1)
        tramo = keypoints[corte]
        valores, cuentas = np.unique(tareas[corte], return_counts=True)
        tarea = str(valores[cuentas.argmax()])

        analisis = None
        if _is_lift(tarea) and len(tramo) >= 2:
            analisis = asdict(
                analyze_lift(
                    tramo,
                    HZ,
                    worker_height_cm=config.worker_height_cm,
                    load_kg=config.load_kg,
                    coupling=config.coupling,
                    asymmetry_deg=45.0 if config.task_requires_twist else 0.0,
                    lifts_per_min=por_minuto,
                )
            ) | {"frequency_estimated": frecuencia_estimada, "frequency_measured": round(medida, 1)}

        replays.append(
            EventReplay(
                id=f"{config.id}-{subject_data.index:02d}-{n:03d}",
                workstation=config.id,
                shift=f"turno-{subject_data.index:02d}",
                start_seconds=evento.start_seconds,
                duration_seconds=evento.duration_seconds,
                peak_reba=evento.peak_reba,
                peak_level=evento.peak_level,
                dominant_component=evento.dominant_component,
                task=tarea,
                keypoints=np.round(_normalize_for_canvas(tramo), 4).tolist(),
                reba=[int(v) for v in scores["reba"][corte]],
                components=[
                    {c: int(scores[c][i]) for c in ("trunk", "neck", "legs", "upper_arm", "lower_arm")}
                    for i in range(evento.start_frame, evento.end_frame + 1)
                ],
                niosh=analisis,
            )
        )

    return {
        "shift": f"turno-{subject_data.index:02d}",
        "subject": int(subject_data.index),
        "seconds": resumen.measured_seconds,
        "seconds_at_risk": resumen.seconds_at_risk,
        "seconds_by_level": resumen.seconds_by_level,
        "tasks": [asdict(t) for t in informe.tasks],
        "recommendation": informe.recommendation(),
        "events": [asdict(r) for r in replays],
        "tareas_por_fotograma": tareas,
        "scores": scores,
    }


def _worst_lift(eventos: list[dict]) -> dict | None:
    """El levantamiento con peor índice. Es lo que resume el puesto en kilos."""
    con_niosh = [e for e in eventos if e.get("niosh") and e["niosh"].get("lifting_index")]
    if not con_niosh:
        return None
    peor = max(con_niosh, key=lambda e: e["niosh"]["lifting_index"])
    return {"event_id": peor["id"], "task": peor["task"], **peor["niosh"]}


def build_workstation(config: WorkstationConfig, shifts: list[dict]) -> dict:
    """Agrega los turnos de un puesto en lo que se ve en la pantalla principal."""
    scores = {
        clave: np.concatenate([t["scores"][clave] for t in shifts]) for clave in shifts[0]["scores"]
    }
    tareas = np.concatenate([t["tareas_por_fotograma"] for t in shifts])
    informe = build_report(tareas, scores, HZ, config)

    por_nivel: dict[str, float] = {}
    for turno in shifts:
        for nivel, segundos in turno["seconds_by_level"].items():
            por_nivel[nivel] = por_nivel.get(nivel, 0.0) + segundos

    # Todos los eventos cuentan para el total; solo los más graves viajan con su
    # replay. Un panel real pagina igual, y mandar al navegador el esqueleto de
    # cuatrocientos eventos para enseñar doce es peso que nadie mira.
    eventos = [e for turno in shifts for e in turno["events"]]
    eventos.sort(key=lambda e: (-e["peak_reba"], -e["duration_seconds"]))
    total_eventos = len(eventos)

    # El peor levantamiento se busca entre TODOS los eventos, ANTES de recortar. Los
    # doce que conservan replay están elegidos por REBA, y el levantamiento más grave
    # según NIOSH no tiene por qué estar entre ellos: son dos normas que miden cosas
    # distintas, que es justo el motivo de tener las dos. Buscarlo después del
    # recorte dejaba sin índice a puestos que sí tenían peso declarado.
    peor_levantamiento = _worst_lift(eventos)
    eventos = eventos[:MAX_REPLAYS]

    return {
        "id": config.id,
        "name": config.name,
        "configured": config.is_configured,
        "config": {
            "load_kg": config.load_kg,
            "coupling": config.coupling,
            "risk_threshold": config.risk_threshold,
            "min_event_seconds": config.min_event_seconds,
        },
        "shifts": len(shifts),
        "seconds": sum(t["seconds"] for t in shifts),
        "seconds_at_risk": sum(t["seconds_at_risk"] for t in shifts),
        "seconds_by_level": por_nivel,
        "tasks": [asdict(t) for t in informe.tasks],
        "recommendation": informe.recommendation(),
        "is_lower_bound": informe.is_lower_bound,
        "events": eventos,
        "total_events": total_eventos,
        "worst_lift": peor_levantamiento,
        "level_of": {
            str(v): action_level(int(v)) for v in np.unique(scores["reba"]) if int(v) > 0
        },
    }
