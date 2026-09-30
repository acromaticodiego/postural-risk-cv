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

from ..baseline.reba import action_level, reba_from_keypoints
from ..datasets.uwiom import LABEL_FIELDS
from ..models.tcn import SkeletonTCN, normalize_windows
from ..pose.schema import N_JOINTS
from ..product.workstation import COMPONENTS, WorkstationConfig

RAIZ = Path(__file__).resolve().parents[2]
MODELO = RAIZ / "artifacts/modelo"

TARGET_HZ = 10.0
PREVIEW_WIDTH = 480


@dataclass
class LiveEvent:
    start_seconds: float
    duration_seconds: float
    peak_reba: int
    peak_level: str
    dominant_component: str
    task: str


class LiveSession:
    """Una cámara, un puesto, y el sistema entero corriendo encima."""

    def __init__(
        self,
        config: WorkstationConfig,
        camera: int = 0,
        model_dir: Path = MODELO,
        device: str | None = None,
    ):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        meta = json.loads((model_dir / "tcn.json").read_text(encoding="utf-8"))
        self.classes = meta["clases"]
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
        self._streak_start = 0.0
        self._last_task = "—"

    # --- ciclo -------------------------------------------------------------

    def read(self) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Un fotograma: devuelve (imagen, keypoints, caja) o (None, None, None)."""
        ok, frame = self.capture.read()
        if not ok:
            return None, None, None
        resultado = self.pose.predict(frame, device=self.device, verbose=False)[0]
        cajas = None if resultado.boxes is None else resultado.boxes.xyxy.cpu().numpy()
        if cajas is None or not len(cajas):
            return frame, None, None
        areas = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
        slot = int(areas.argmax())
        return frame, resultado.keypoints.data.cpu().numpy()[slot, :, :2], cajas[slot]

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
        if feed is not None:
            keypoints, caja = feed
            frame = None
            want_preview = False
        else:
            frame, keypoints, caja = self.read()
            if frame is None:
                return None

        self.frames += 1
        ahora = self.frames / TARGET_HZ
        presente = keypoints is not None

        if presente:
            self.buffer.append(keypoints)
            datos = reba_from_keypoints(
                keypoints[None, ...], assumptions=self.config.assumptions()
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
        else:
            self.buffer.clear()
            reba, componentes, angulos, nivel = 0, {c: 0 for c in COMPONENTS}, {}, "sin persona"
            self._last_task = "—"

        self._track_event(reba, componentes, ahora)

        salida = {
            "frame": self.frames,
            "seconds": round(ahora, 2),
            "present": presente,
            "reba": reba,
            "level": nivel,
            "components": componentes,
            "angles": angulos,
            "task": self._last_task,
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

    def _track_event(self, reba: int, componentes: dict[str, int], ahora: float) -> None:
        """Cierra un tramo de riesgo cuando termina, no mientras dura."""
        if reba >= self.config.risk_threshold:
            if not self._streak:
                self._streak_start = ahora
            peor = max(componentes, key=componentes.get) if componentes else "trunk"
            self._streak.append((reba, peor))
            return
        if self._streak:
            duracion = len(self._streak) / TARGET_HZ
            if duracion >= self.config.min_event_seconds:
                picos = [r for r, _ in self._streak]
                causas = [c for _, c in self._streak]
                pico = max(picos)
                self.events.append(
                    LiveEvent(
                        start_seconds=round(self._streak_start, 1),
                        duration_seconds=round(duracion, 1),
                        peak_reba=pico,
                        peak_level=action_level(pico),
                        dominant_component=max(set(causas), key=causas.count),
                        task=self._last_task,
                    )
                )
            self._streak = []

    def close(self) -> None:
        self.capture.release()
