"""El puesto de trabajo: la entidad que convierte esto en un producto.

Decidido en el ADR 0001. De los componentes de REBA, la cámara no puede medir el
peso de la carga, la calidad del agarre ni la torsión del tronco. La reacción de
laboratorio es asumir valores neutros y publicar una cota inferior; la de producto
es reconocer que **en una planta esos datos existen y el cliente los conoce**. El
ingeniero de seguridad sabe que en la línea 3 las cajas pesan 12 kg. Pedirle a una
red que lo adivine sustituye un dato exacto por una estimación sin intervalo de
confianza.

Así que cada puesto se configura, y con su configuración el puntaje deja de ser
una cota inferior y pasa a ser REBA completo para ese puesto. Sin configurar, se
usan los neutros y el informe lo marca como estimación mínima.

Que añadir una línea de producción sea añadir un puesto, y no tocar código, es lo
que hace la arquitectura escalable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..baseline.reba import ACTION_LEVELS, RebaAssumptions, action_level

# Tramos de carga de la norma, en kilogramos.
LOAD_BANDS = ((5.0, 0), (10.0, 1))
LOAD_SCORE_HEAVY = 2

COUPLING_SCORES = {"good": 0, "fair": 1, "poor": 2, "unacceptable": 3}

# Valores por defecto del ADR 0001. Son parámetros, no constantes escondidas: un
# cliente con otro criterio los cambia, y todo número publicado dice con qué
# valores se obtuvo.
DEFAULT_RISK_THRESHOLD = 4
DEFAULT_MIN_EVENT_SECONDS = 1.0

COMPONENTS = ("trunk", "neck", "legs", "upper_arm", "lower_arm")


@dataclass(frozen=True)
class WorkstationConfig:
    """Lo que el cliente declara de un puesto, y la cámara no tiene que adivinar."""

    id: str
    name: str
    load_kg: float | None = None
    coupling: str | None = None
    task_requires_twist: bool = False
    sudden_load: bool = False
    risk_threshold: int = DEFAULT_RISK_THRESHOLD
    min_event_seconds: float = DEFAULT_MIN_EVENT_SECONDS
    worker_height_cm: float | None = None
    """Estatura del trabajador, para convertir píxeles en centímetros y poder
    aplicar la ecuación NIOSH. El cuerpo hace de regla, que es lo que hace un
    evaluador midiendo sobre una foto. Sin declararla se asume una media y el
    resultado se marca como estimado."""

    lifts_per_min: float | None = None
    """Levantamientos por minuto SOSTENIDOS en el puesto. Se declara y no se mide,
    y esto costó un número inflado antes de entenderlo: NIOSH supone que la
    frecuencia se mantiene durante horas, así que contarla sobre tres minutos de
    grabación —donde alguien levanta sin parar porque le están grabando— da una
    frecuencia de experimento y hunde el peso recomendado. Un jefe de planta sabe
    cuántas cajas por hora salen de su línea; la cámara solo sabe lo que vio en la
    muestra. Sin declarar, se usa la medida y el informe lo marca como estimada."""

    def __post_init__(self) -> None:
        if self.coupling is not None and self.coupling not in COUPLING_SCORES:
            raise ValueError(f"agarre desconocido: {self.coupling!r}")
        if self.load_kg is not None and self.load_kg < 0:
            raise ValueError(f"carga negativa: {self.load_kg}")
        if not 1 <= self.risk_threshold <= 15:
            raise ValueError(f"umbral de riesgo fuera de la escala: {self.risk_threshold}")

    @property
    def is_configured(self) -> bool:
        """Si falta algo, el puntaje es una cota inferior y hay que decirlo."""
        return self.load_kg is not None and self.coupling is not None

    def load_score(self) -> int:
        if self.load_kg is None:
            return 0
        base = LOAD_SCORE_HEAVY
        for limite, puntaje in LOAD_BANDS:
            if self.load_kg < limite:
                base = puntaje
                break
        return min(base + int(self.sudden_load), 3)

    def assumptions(self) -> RebaAssumptions:
        """Los componentes no visibles, tomados de la configuración del puesto."""
        return RebaAssumptions(
            wrist=1,  # sigue sin ser observable: el esqueleto acaba en la muñeca
            trunk_twist=self.task_requires_twist,
            neck_twist=False,
            load_force=self.load_score(),
            coupling=COUPLING_SCORES.get(self.coupling or "good", 0),
        )


@dataclass(frozen=True)
class RiskEvent:
    """Un tramo de riesgo sostenido, con lo que hace falta para actuar sobre él."""

    start_frame: int
    end_frame: int
    start_seconds: float
    duration_seconds: float
    peak_reba: int
    peak_level: str
    dominant_component: str
    """Qué articulación empujó el puntaje. Sin esto el informe dice «riesgo alto»
    y nadie sabe qué rediseñar; con esto dice «el tronco, recogiendo del suelo»."""


@dataclass(frozen=True)
class ExposureSummary:
    """Lo que se le entrega a un ingeniero de SST al final de un turno."""

    workstation: str
    seconds_by_level: dict[str, float]
    events: list[RiskEvent]
    measured_seconds: float
    seconds_without_person: float
    is_lower_bound: bool
    """True si el puesto no estaba configurado: entonces el riesgo real es MAYOR o
    igual al reportado, y ningún puesto sin alertas se puede leer como seguro."""
    threshold: int
    min_event_seconds: float = field(default=DEFAULT_MIN_EVENT_SECONDS)

    @property
    def seconds_at_risk(self) -> float:
        return sum(
            s
            for nivel, s in self.seconds_by_level.items()
            if nivel not in ("despreciable", "bajo", "sin persona")
        )


def _dominant_component(components: dict[str, np.ndarray], corte: slice) -> str:
    """El componente con el puntaje medio más alto en el tramo.

    Se usa la media y no el máximo porque un pico de un fotograma puede ser un
    fallo del detector, mientras que un componente alto durante todo el tramo es
    lo que de verdad caracteriza la postura.
    """
    medias = {nombre: float(components[nombre][corte].mean()) for nombre in COMPONENTS}
    return max(medias, key=medias.get)


def summarize_exposure(
    scores: dict[str, np.ndarray],
    hz: float,
    config: WorkstationConfig,
    present: np.ndarray | None = None,
) -> ExposureSummary:
    """Convierte puntajes por fotograma en el informe del turno.

    Las dos caras del ADR 0001 no se mezclan:

      · la EXPOSICIÓN ACUMULADA suma todo fotograma de riesgo, sin mínimo de
        duración. Filtrarla sesgaría el informe a la baja y no al azar: se caerían
        las posturas breves y repetidas, que son el mecanismo de la lesión por
        esfuerzo repetitivo.
      · los EVENTOS exigen un segundo sostenido, porque son lo que dispara una
        alerta y una alerta falsa hace que el sistema se desconecte.
    """
    reba = np.asarray(scores["reba"], dtype=int)
    n = len(reba)
    if present is None:
        present = reba > 0
    dt = 1.0 / hz

    # --- exposición acumulada, sin filtrar ---
    segundos: dict[str, float] = {nombre: 0.0 for _, _, nombre in ACTION_LEVELS}
    segundos["sin persona"] = float((~present).sum()) * dt
    for puntaje in np.unique(reba[present]):
        nivel = action_level(int(puntaje))
        segundos[nivel] += float((reba[present] == puntaje).sum()) * dt

    # --- eventos: tramos sostenidos por encima del umbral ---
    en_riesgo = present & (reba >= config.risk_threshold)
    minimo = max(1, int(round(config.min_event_seconds * hz)))
    eventos: list[RiskEvent] = []
    inicio = None
    for i in range(n + 1):
        activo = bool(en_riesgo[i]) if i < n else False
        if activo and inicio is None:
            inicio = i
        elif not activo and inicio is not None:
            largo = i - inicio
            if largo >= minimo:
                corte = slice(inicio, i)
                pico = int(reba[corte].max())
                eventos.append(
                    RiskEvent(
                        start_frame=inicio,
                        end_frame=i - 1,
                        start_seconds=inicio * dt,
                        duration_seconds=largo * dt,
                        peak_reba=pico,
                        peak_level=action_level(pico),
                        dominant_component=_dominant_component(scores, corte),
                    )
                )
            inicio = None

    return ExposureSummary(
        workstation=config.id,
        seconds_by_level=segundos,
        events=eventos,
        measured_seconds=n * dt,
        seconds_without_person=segundos["sin persona"],
        is_lower_bound=not config.is_configured,
        threshold=config.risk_threshold,
        min_event_seconds=config.min_event_seconds,
    )
