"""El sistema funcionando en vivo sobre una cámara.

Mismo pipeline que el de lotes, con tres diferencias que importan:

1. **Se procesa a 10 Hz aunque la cámara dé 30.** El modelo se entrenó con ventanas
   de 2 segundos a 10 Hz, o sea 20 fotogramas. Si aquí se le dieran 20 fotogramas
   de una cámara a 30 fps, estaría viendo 0,67 segundos de movimiento y creyendo
   que son dos: el gesto le llegaría acelerado tres veces. No daría ningún error,
   solo predicciones peores sin motivo aparente. La tasa de proceso es parte del
   contrato del modelo, no un parámetro de rendimiento.

2. **Los eventos se cierran en caliente.** En lotes se conoce la secuencia entera;
   aquí un tramo de riesgo solo se puede confirmar cuando termina, así que se lleva
   una racha y se emite el evento al romperse, si duró lo suficiente.

3. **La imagen de la cámara se emite pero NO SE GUARDA EN NINGÚN SITIO.** Es la
   vista de instalación: quien monta la cámara necesita ver el encuadre, y quien ve
   la demo necesita entender la diferencia entre lo que la cámara capta y lo único
   que sale del dispositivo. El fotograma vive lo que dura el envío por el socket;
   no hay ningún camino por el que llegue a disco.
"""

from __future__ import annotations

import base64
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from ..baseline.niosh import _pixels_per_cm, analyze_lift
from ..baseline.reba import action_level, reba_from_keypoints
from ..baseline.tecnica import VERDICTS, assess_technique
from ..datasets.uwiom import LABEL_FIELDS
from ..load.carga import associate_to_person, build_load
from ..models.tcn import SkeletonTCN, normalize_windows
from ..pose.schema import N_JOINTS
from ..product.workstation import COMPONENTS, WorkstationConfig

RAIZ = Path(__file__).resolve().parents[2]
MODELO = RAIZ / "artifacts/modelo"

TARGET_HZ = 10.0
PREVIEW_WIDTH = 480

# Cada cuántos fotogramas se busca la carga. Dos modelos en la ruta crítica no caben
# en 100 ms: la pose sola cuesta ~74 ms en una RTX 3050 y el detector de carga añade
# lo suyo. Una caja en las manos no se mueve de sitio en 300 ms —va pegada a la
# persona, que sí se sigue a 10 Hz— así que se busca cada tres fotogramas y entre
# medias se conserva la última. Bajar la tasa de la pose en su lugar sería peor:
# rompería la ventana de 2 s con la que se entrenó el modelo de tareas.
LOAD_EVERY = 3


@dataclass
class LiveEvent:
    start_seconds: float
    duration_seconds: float
    peak_reba: int
    peak_level: str
    dominant_component: str
    task: str
    niosh: dict | None = None
    """Solo si el evento fue un LEVANTAMIENTO. La ecuación mide levantar una carga,
    no estar de pie en mala postura."""


# Un evento es un levantamiento si su tarea lo dice. Igual que en el modo por lotes.
LIFTING_TASKS = ("pick-up", "place")


class LiveSession:
    """Una cámara, un puesto, y el sistema entero corriendo encima."""

    def __init__(
        self,
        config: WorkstationConfig,
        camera: int = 0,
        model_dir: Path = MODELO,
        device: str | None = None,
        load_weights: str | Path | None = None,
    ):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        meta = json.loads((model_dir / "tcn.json").read_text(encoding="utf-8"))
        self.classes = meta["clases"]
        self.view = meta["vista"]
        self.window_frames = int(round(meta["window_seconds"] * meta["hz"]))
        self.hz = float(meta["hz"])
        if abs(self.hz - TARGET_HZ) > 1e-6:
            raise ValueError(
                f"el modelo se entrenó a {self.hz} Hz y la sesión procesa a {TARGET_HZ}: "
                "las ventanas no serían equivalentes"
            )

        self.model = SkeletonTCN({c: len(self.classes[c]) for c in LABEL_FIELDS}).to(self.device)
        self.model.load_state_dict(torch.load(model_dir / "tcn.pt", map_location=self.device))
        self.model.eval()

        # `camera=None` construye la sesión SIN abrir ninguna cámara, para poder
        # alimentarla con esqueletos y comprobar todo lo que solo ocurre cuando hay
        # una persona delante: la predicción de la tarea, el puntaje y el cierre de
        # eventos. Sin esto, esa mitad del modo en vivo solo se podría probar con
        # alguien plantado frente al portátil.
        self.pose = None
        self.capture = None
        if camera is not None:
            from ultralytics import YOLO

            self.pose = YOLO("yolo11n-pose.pt")
            self.capture = cv2.VideoCapture(camera)
            if not self.capture.isOpened():
                raise RuntimeError(f"no se pudo abrir la cámara {camera}")

        self.buffer: deque[np.ndarray] = deque(maxlen=self.window_frames)
        self.started = time.perf_counter()
        self.frames = 0
        self.seconds_by_level: dict[str, float] = {}
        self.events: list[LiveEvent] = []
        self._streak: list[tuple[int, str]] = []
        self._streak_poses: list[np.ndarray] = []
        self._streak_tasks: list[str] = []
        self._streak_loads: list[np.ndarray | None] = []

        # El detector de carga es opcional: sin él, el sistema funciona igual y
        # NIOSH mide a las muñecas en vez de al centro de la caja.
        self.load_detector = None
        if load_weights and Path(load_weights).exists():
            from ..load.carga import LoadDetector

            self.load_detector = LoadDetector(str(load_weights), device=self.device)
        self._last_load = None
        self._streak_start = 0.0
        self._last_task = "—"

    # --- ciclo -------------------------------------------------------------

    def read(self):
        """Un fotograma: (imagen, keypoints, caja, confianzas) o todo None."""
        ok, frame = self.capture.read()
        if not ok:
            return None, None, None, None
        resultado = self.pose.predict(frame, device=self.device, verbose=False)[0]
        cajas = None if resultado.boxes is None else resultado.boxes.xyxy.cpu().numpy()
        if cajas is None or not len(cajas):
            return frame, None, None, None
        areas = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
        slot = int(areas.argmax())
        kp = resultado.keypoints.data.cpu().numpy()[slot]
        return frame, kp[:, :2], cajas[slot], kp[:, 2]

    def check_trust(self, task: str, trunk_deg: float, keypoints: np.ndarray) -> str | None:
        """¿Hay motivo para no fiarse de la tarea que acaba de predecir el modelo?

        Nace de una prueba real (30/09): con la cámara de un portátil en un salón, el
        modelo contestó `bend` mientras la geometría medía 2 grados de tronco. El
        puntaje de riesgo estaba bien —REBA no depende del modelo— pero la tarea era
        imposible, y el panel la mostraba con la misma seguridad que las demás.

        Se comprueban dos cosas independientes:

          · **Coherencia.** La tarea y los ángulos salen de dos caminos distintos: uno
            aprendido y otro geométrico. Cuando se contradicen, al menos uno se
            equivoca, y el geométrico es el que está atado a la norma.
          · **Dominio de validez.** La razón entre anchura de hombros y altura del
            cuerpo describe el ángulo de cámara. Fuera del rango que el modelo vio
            entrenando está extrapolando, y un modelo que extrapola contesta con la
            misma seguridad que cuando sabe.

        Un sistema que sabe cuándo no sabe vale más que uno que acierta un poco más.
        """
        from ..pose.schema import JOINT

        alto = max(float(keypoints[:, 1].max() - keypoints[:, 1].min()), 1.0)
        razon = abs(
            float(keypoints[JOINT["left_shoulder"], 0] - keypoints[JOINT["right_shoulder"], 0])
        ) / alto
        if not (self.view["shoulder_ratio_p01"] <= razon <= self.view["shoulder_ratio_p99"]):
            return (
                f"angulo de camara fuera de lo entrenado (hombros/altura {razon:.2f}, "
                f"visto {self.view['shoulder_ratio_p01']:.2f}-{self.view['shoulder_ratio_p99']:.2f}): "
                "la tarea no es fiable"
            )
        if task.startswith("bend") and trunk_deg < 20:
            return f"el modelo dice agachado y el tronco mide {trunk_deg:.0f} grados: no cuadra"
        if task.startswith("stand") and trunk_deg > 50:
            return f"el modelo dice de pie y el tronco mide {trunk_deg:.0f} grados: no cuadra"
        return None

    def predict_task(self) -> str:
        """La tarea, en cuanto hay ventana completa. Antes, no se inventa nada."""
        if len(self.buffer) < self.window_frames:
            return "esperando contexto"
        ventana = normalize_windows(np.stack(self.buffer)[None, ...])
        with torch.no_grad():
            salida = self.model(torch.tensor(ventana, device=self.device))
        campos = {
            c: self.classes[c][int(salida[c].argmax(dim=1)[0])] for c in LABEL_FIELDS
        }
        return f"{campos['motion']} / {campos['manipulation']} / {campos['height']}"

    def step(
        self,
        want_preview: bool = True,
        feed: tuple[np.ndarray | None, np.ndarray | None] | None = None,
    ) -> dict | None:
        """Procesa un fotograma y devuelve lo que se pinta en pantalla.

        `feed` permite inyectar (keypoints, caja) en vez de leer la cámara. Es lo
        que usan las pruebas para ejercitar todo el camino de una persona presente
        sin necesitar a nadie delante del portátil.
        """
        confianzas = None
        if feed is not None:
            keypoints, caja = feed
            frame = None
            want_preview = False
        else:
            frame, keypoints, caja, confianzas = self.read()
            if frame is None:
                return None

        self.frames += 1
        ahora = self.frames / TARGET_HZ
        presente = keypoints is not None

        if presente:
            self.buffer.append(keypoints)
            self._detect_load(frame, keypoints)
            datos = reba_from_keypoints(
                keypoints[None, ...],
                assumptions=self.config.assumptions(),
                scores=None if confianzas is None else confianzas[None, ...],
            )
            reba = int(datos["reba"][0])
            componentes = {c: int(datos[c][0]) for c in COMPONENTS}
            angulos = {
                k: round(float(v[0]), 1) for k, v in datos.items() if k.endswith(("flexion", "elevation"))
            }
            nivel = action_level(reba)
            self.seconds_by_level[nivel] = self.seconds_by_level.get(nivel, 0.0) + 1 / TARGET_HZ
            # La tarea se recalcula cada tres fotogramas para no gastar GPU de más,
            # PERO también en cuanto lo que hay no es una tarea de verdad: si no,
            # al llenarse la ventana el panel se quedaba diciendo «esperando
            # contexto» hasta que tocara el siguiente múltiplo de tres.
            if self.frames % 3 == 0 or "/" not in self._last_task:
                self._last_task = self.predict_task()
            aviso = (
                self.check_trust(self._last_task, angulos.get("trunk_flexion", 0.0), keypoints)
                if "/" in self._last_task
                else None
            )
        else:
            self.buffer.clear()
            reba, componentes, angulos, nivel = 0, {c: 0 for c in COMPONENTS}, {}, "sin persona"
            self._last_task = "—"
            aviso = None
            self._last_load = None

        self._track_event(reba, componentes, ahora, keypoints if presente else None)

        salida = {
            "frame": self.frames,
            "seconds": round(ahora, 2),
            "present": presente,
            "reba": reba,
            "level": nivel,
            "components": componentes,
            "angles": angulos,
            "task": self._last_task,
            "task_warning": aviso,
            "partial": bool(datos["partial"][0]) if presente else False,
            "unreliable": (
                [c for c, ok in datos["reliable"].items() if not bool(ok[0])] if presente else []
            ),
            "side": str(datos["side"][0]) if presente else None,
            "frontal_view": bool(datos["frontal_view"][0]) if presente else False,
            "technique": self._technique(datos) if presente else None,
            "load": self._load_payload(),
            "skeleton": self._canvas_skeleton(keypoints, caja) if presente else None,
            "seconds_by_level": {k: round(v, 1) for k, v in self.seconds_by_level.items()},
            "events": [e.__dict__ for e in self.events[-8:]],
            "total_events": len(self.events),
            "threshold": self.config.risk_threshold,
            "workstation": self.config.name,
            "configured": self.config.is_configured,
        }
        if want_preview:
            salida["preview"] = self._preview(frame, keypoints)
        return salida

    # --- piezas ------------------------------------------------------------

    def _canvas_skeleton(self, keypoints: np.ndarray, caja: np.ndarray) -> list:
        """El esqueleto en 0..1, encajado en su propia caja y con escala isotrópica.

        Isotrópica, igual que en el entrenamiento: dividir cada eje por su lado de
        la caja deformaría los ángulos, y los ángulos son lo que se está midiendo.
        """
        alto = max(float(caja[3] - caja[1]), 1.0)
        centro = np.array([(caja[0] + caja[2]) / 2, (caja[1] + caja[3]) / 2])
        return np.round((keypoints - centro) / alto * 0.8 + 0.5, 4).tolist()

    def _preview(self, frame: np.ndarray, keypoints: np.ndarray | None) -> str:
        """La imagen de la cámara con el esqueleto encima, para la vista de montaje.

        Se reduce y se comprime porque viaja por un socket diez veces por segundo, y
        porque nadie necesita resolución completa para comprobar un encuadre.
        """
        escala = PREVIEW_WIDTH / frame.shape[1]
        pequeno = cv2.resize(frame, (PREVIEW_WIDTH, int(frame.shape[0] * escala)))
        if keypoints is not None:
            from ..viz.render import HUESOS
            from ..pose.schema import JOINT

            puntos = keypoints * escala
            for a, b in HUESOS:
                pa, pb = puntos[JOINT[a]], puntos[JOINT[b]]
                if (pa == 0).all() or (pb == 0).all():
                    continue
                cv2.line(pequeno, tuple(pa.astype(int)), tuple(pb.astype(int)), (120, 220, 90), 2, cv2.LINE_AA)
            for p in puntos:
                if (p == 0).all():
                    continue
                cv2.circle(pequeno, tuple(p.astype(int)), 3, (120, 220, 90), -1, cv2.LINE_AA)
        ok, buffer = cv2.imencode(".jpg", pequeno, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buffer).decode("ascii") if ok else ""

    def _detect_load(self, frame, keypoints: np.ndarray) -> None:
        """Busca la carga y decide si está en las manos.

        Solo cada `LOAD_EVERY` fotogramas, y entre medias se conserva la anterior:
        dos modelos no caben en los 100 ms del ciclo, y una caja sostenida no se
        mueve de las manos en 300 ms.
        """
        if self.load_detector is None or frame is None:
            return
        if self.frames % LOAD_EVERY != 0 and self._last_load is not None:
            return
        cajas, confianzas, poligonos = self.load_detector.detect(frame)
        indice = associate_to_person(cajas, keypoints)
        if indice is None:
            self._last_load = None
            return
        px_cm = None
        if self.config.worker_height_cm:
            px_cm = _pixels_per_cm(keypoints, self.config.worker_height_cm)
        self._last_load = build_load(
            cajas[indice],
            float(confianzas[indice]),
            px_cm,
            tuple(self.config.load_catalog) or None,
            poligonos[indice] if indice < len(poligonos) else None,
        )

    def _technique(self, datos) -> dict:
        """El consejo de técnica: qué hacer distinto, o que ya está bien hecho.

        Existe porque el puntaje solo no sirve de consejo: agacharse doblando la
        espalda y agacharse en cuclligas dan el MISMO REBA (medido: 4 y 4), y uno
        es evitable y el otro es el mínimo de la tarea.
        """
        codigo = str(
            assess_technique(
                datos["trunk"], datos["legs"], datos["upper_arm"], datos["reba"],
                threshold=self.config.risk_threshold,
            )[0]
        )
        v = VERDICTS[codigo]
        return {"code": v.code, "message": v.message, "avoidable": v.avoidable}

    def _load_payload(self) -> dict | None:
        if self._last_load is None:
            return None
        c = self._last_load
        return {
            "box": c.box,
            "center": c.center,
            "confidence": c.confidence,
            "width_cm": c.width_cm,
            "matched": c.matched.name if c.matched else None,
            "weight_kg": c.weight_kg,
        }

    @property
    def _effective_load_kg(self) -> float | None:
        """El peso que se aplica: el de la carga vista si el catálogo la reconoce, y
        si no el del puesto. Es el salto de «una constante» a «lo que hay en las
        manos»."""
        if self._last_load is not None and self._last_load.weight_kg is not None:
            return self._last_load.weight_kg
        return self.config.load_kg

    def _track_event(
        self,
        reba: int,
        componentes: dict[str, int],
        ahora: float,
        keypoints: np.ndarray | None = None,
    ) -> None:
        """Cierra un tramo de riesgo cuando termina, no mientras dura."""
        if reba >= self.config.risk_threshold:
            if not self._streak:
                self._streak_start = ahora
                self._streak_poses = []
                self._streak_tasks = []
                self._streak_loads = []
            peor = max(componentes, key=componentes.get) if componentes else "trunk"
            self._streak.append((reba, peor))
            if keypoints is not None:
                self._streak_poses.append(keypoints)
            self._streak_tasks.append(self._last_task)
            self._streak_loads.append(
                np.array(self._last_load.center) if self._last_load else None
            )
            return
        if self._streak:
            duracion = len(self._streak) / TARGET_HZ
            if duracion >= self.config.min_event_seconds:
                picos = [r for r, _ in self._streak]
                causas = [c for _, c in self._streak]
                pico = max(picos)
                # La tarea del evento es la DOMINANTE durante el tramo, no la del
                # fotograma que lo cierra: ese es justo aquel en el que el riesgo ya
                # bajó, y su tarea suele ser otra. Es como se hace en el modo por
                # lotes, y aquí estaba mal.
                candidatas = [t for t in self._streak_tasks if "/" in t]
                tarea_evento = (
                    max(set(candidatas), key=candidatas.count) if candidatas else self._last_task
                )
                # NIOSH solo si el tramo fue un levantamiento y hay esqueletos con
                # los que medir la distancia de la carga al cuerpo.
                analisis = None
                if (
                    any(t in tarea_evento for t in LIFTING_TASKS)
                    and len(self._streak_poses) >= 2
                ):
                    # Los centros de la carga solo se pasan si se vio en TODOS los
                    # fotogramas del tramo: una lista a medias mezclaría la posición
                    # de la caja en unos instantes con la de las muñecas en otros, y
                    # la distancia resultante no sería de ningún momento concreto.
                    centros = (
                        np.stack(self._streak_loads)
                        if self._streak_loads
                        and len(self._streak_loads) == len(self._streak_poses)
                        and all(c is not None for c in self._streak_loads)
                        else None
                    )
                    analisis = analyze_lift(
                        np.stack(self._streak_poses),
                        TARGET_HZ,
                        worker_height_cm=self.config.worker_height_cm,
                        load_kg=self._effective_load_kg,
                        coupling=self.config.coupling,
                        asymmetry_deg=45.0 if self.config.task_requires_twist else 0.0,
                        lifts_per_min=self.config.lifts_per_min or 2.0,
                        load_centers=centros,
                    ).__dict__
                self.events.append(
                    LiveEvent(
                        start_seconds=round(self._streak_start, 1),
                        duration_seconds=round(duracion, 1),
                        peak_reba=pico,
                        peak_level=action_level(pico),
                        dominant_component=max(set(causas), key=causas.count),
                        task=tarea_evento,
                        niosh=analisis,
                    )
                )
            self._streak = []
            self._streak_poses = []
            self._streak_tasks = []
            self._streak_loads = []

    def close(self) -> None:
        self.capture.release()
