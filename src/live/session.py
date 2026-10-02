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
# Ancho al que viaja la imagen de la camara. 480 bastaba cuando el panel era un
# recuadro; con la consola ocupando la pantalla, el navegador la estaba AMPLIANDO y
# se veia blanda. Medido sobre fotogramas reales: a 720 px el redimensionado mas el
# JPEG cuestan 7,4 ms y 387 KB/s a 10 Hz, dentro de un ciclo que usa 34 de 100 ms.
PREVIEW_WIDTH = 720

# Cuantas medidas de escala se guardan para quedarse con la mayor. A 10 Hz y una
# busqueda de carga cada 3 fotogramas, 20 son unos 6 segundos: lo que tarda un
# levantamiento completo, y poco para que alguien cambie de distancia a la camara.
ESCALA_VENTANA = 20

# Cada cuántos fotogramas se busca la carga. Dos modelos en la ruta crítica no caben
# en 100 ms: la pose sola cuesta ~74 ms en una RTX 3050 y el detector de carga añade
# lo suyo. Una caja en las manos no se mueve de sitio en 300 ms —va pegada a la
# persona, que sí se sigue a 10 Hz— así que se busca cada tres fotogramas y entre
# medias se conserva la última. Bajar la tasa de la pose en su lugar sería peor:
# rompería la ventana de 2 s con la que se entrenó el modelo de tareas.
LOAD_EVERY = 3

# Cuantos fotogramas seguidos por debajo del umbral se toleran DENTRO de una racha
# antes de darla por terminada.
#
# Sin esto el sistema no cerraba ni un evento en un video de alguien levantando
# cajas, y no por no ver el riesgo: 65 de 128 fotogramas pasaban de REBA 4. El
# puntaje parpadea entre 3 y 4 —"6 4 4 4 3 4 4 3 4 4 3 6 3 4..."— y un solo
# fotograma en 3 reiniciaba la racha entera, asi que la mas larga duraba 0,9 s
# contra el 1,0 s que pide la norma de este sistema. Fallaba por un fotograma.
#
# Medido sobre esa serie: con 0 de tolerancia salen 0 eventos, con 1 sale 1, y con 2
# salen 2. Se eligen 2 (0,2 s) porque un fotograma suelto por debajo es temblor del
# detector de pose, no que la persona se haya erguido y vuelto a agachar en una
# decima. Pasar de ahi empieza a fundir levantamientos distintos en uno.
#
# El tiempo tolerado SI cuenta en la duracion del evento, porque la postura de
# riesgo no se interrumpio de verdad: lo que fallo fue la medida.
EVENT_GAP_FRAMES = 2

# Cuantas medidas de anchura se guardan para quedarse con la mayor plausible.
#
# La camara no mide la caja: mide su PROYECCION, y eso solo coincide con la anchura
# real cuando la cara de la caja esta paralela al plano de la imagen. Medido sobre 67
# fotogramas de una grabacion real, el mismo objeto se proyecta entre 96 y 231
# pixeles segun como este girado — 2,4x— y el panel llegaba a publicar "25 cm" de una
# caja que pasa de 40, lo cual era CIERTO como proyeccion y falso como anchura.
#
# La proyeccion mas ancha de una caja es su anchura real, asi que de la ventana se
# toma el percentil 90 y no el maximo: con el maximo, una sola deteccion demasiado
# grande se queda mandando los seis segundos siguientes.
ANCHO_VENTANA = 20


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
        self._descartar = 0
        if camera is not None:
            from ultralytics import YOLO

            self.pose = YOLO("yolo11n-pose.pt")
            self.capture = cv2.VideoCapture(camera)
            if not self.capture.isOpened():
                raise RuntimeError(f"no se pudo abrir la cámara {camera}")
            # El ciclo consume 10 fotogramas por segundo y la cámara produce 30: los
            # otros 20 se quedan en el búfer del driver, que es una COLA. A los diez
            # segundos se van doscientos fotogramas de retraso y lo que se ve en
            # pantalla es lo que pasó hace siete segundos. No es lentitud de cómputo
            # —el ciclo mide 34 ms de 100 disponibles— sino latencia acumulada, y se
            # nota exactamente igual.
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            fps_camara = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
            self._descartar = max(0, int(round(fps_camara / TARGET_HZ)) - 1)

        self.buffer: deque[np.ndarray] = deque(maxlen=self.window_frames)
        self.started = time.perf_counter()
        self.frames = 0
        self.seconds_by_level: dict[str, float] = {}
        self.events: list[LiveEvent] = []
        self._streak: list[tuple[int, str]] = []
        self._streak_gap = 0
        self._streak_poses: list[np.ndarray] = []
        self._streak_tasks: list[str] = []
        self._streak_loads: list[np.ndarray | None] = []

        # El detector de carga es opcional: sin él, el sistema funciona igual y
        # NIOSH mide a las muñecas en vez de al centro de la caja.
        #
        # Y si no se dice cuál, gana el de la planta sobre el de fábrica. Ese orden ES
        # la decisión de producto del ADR 0002 escrita en una línea de código: el
        # modelo viene de fábrica y cada cliente lo calibra con sus cargas, así que en
        # cuanto existe un modelo calibrado para esta planta, es el que manda.
        if load_weights is None:
            for candidato in (model_dir / "carga-cliente.pt", model_dir / "carga.pt"):
                if candidato.exists():
                    load_weights = candidato
                    break

        self.load_detector = None
        self.load_weights: Path | None = None
        if load_weights and Path(load_weights).exists():
            from ..load.carga import LoadDetector

            self.load_detector = LoadDetector(str(load_weights), device=self.device)
            self.load_weights = Path(load_weights)
        self._escalas: list[float] = []
        self._anchos: list[float] = []
        self._last_load = None
        self._streak_start = 0.0
        self._last_task = "—"

    # --- ciclo -------------------------------------------------------------

    def read(self):
        """Un fotograma: (imagen, keypoints, caja, confianzas) o todo None.

        Se descartan los fotogramas que la cámara ha producido mientras se procesaba
        el anterior. `grab()` los saca de la cola sin decodificarlos, que cuesta una
        fracción de lo que cuesta `read()`; sin esto la imagen va acumulando retraso
        indefinidamente y el sistema parece lento cuando lo que está es desfasado.

        Y se descartan en vez de procesarlos porque la tasa de 10 Hz es parte del
        contrato del modelo: la ventana de 2 s con la que se entrenó son 20 fotogramas
        y alimentarlo más deprisa le daría el gesto acelerado.
        """
        for _ in range(self._descartar):
            self.capture.grab()
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
        feed: tuple | None = None,
    ) -> dict | None:
        """Procesa un fotograma y devuelve lo que se pinta en pantalla.

        `feed` permite inyectar (keypoints, caja) —y opcionalmente la imagen— en vez
        de leer la cámara. Es lo que usan las pruebas para ejercitar todo el camino de
        una persona presente sin necesitar a nadie delante del portátil.

        La imagen es opcional y no estaba al principio, y su ausencia dejaba fuera
        justo el detector de carga: sin fotograma no hay nada que detectar, así que esa
        mitad del modo en vivo solo se podía comprobar con una caja en las manos
        delante de la webcam. Con ella, una grabación guardada recorre el mismo camino
        que la cámara.
        """
        confianzas = None
        if feed is not None:
            keypoints, caja = feed[0], feed[1]
            frame = feed[2] if len(feed) > 2 else None
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
            "load": self._load_payload(caja) if presente else None,
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

    def _canvas_load(self, caja: np.ndarray) -> list | None:
        """El contorno de la carga en 0..1, con la MISMA transformación que el esqueleto.

        Tiene que ser la misma o la caja saldría desplazada respecto a las manos que la
        sujetan, que es justo lo que esta vista existe para enseñar. Por eso se calcula
        aquí, donde está la caja de la persona, y no dentro de `_load_payload`.

        Y que la carga aparezca en la vista de esqueleto no es decorado: esa columna es
        «lo único que sale del dispositivo». Si el sistema aplica el peso de una caja
        que vio, la caja tiene que estar en lo que se conserva, o el replay de un
        incidente no podría explicar de dónde salió ese peso.
        """
        if self._last_load is None:
            return None
        alto = max(float(caja[3] - caja[1]), 1.0)
        centro = np.array([(caja[0] + caja[2]) / 2, (caja[1] + caja[3]) / 2])
        contorno = self._last_load.polygon
        if contorno:
            puntos = np.asarray(contorno, dtype=np.float32)
        else:
            x0, y0, x1, y1 = self._last_load.box
            puntos = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        return np.round((puntos - centro) / alto * 0.8 + 0.5, 4).tolist()

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
        # La carga se dibuja DESPUES del esqueleto: al reves, la linea verde del
        # cuerpo cruzaba por encima de la etiqueta y la dejaba a medias leer.
        if self._last_load is not None:
            contorno = self._last_load.polygon
            if contorno:
                puntos = (np.asarray(contorno, dtype=np.float32) * escala).astype(int)
                cv2.polylines(pequeno, [puntos], True, (60, 180, 255), 2, cv2.LINE_AA)
            else:
                x0, y0, x1, y1 = (int(v * escala) for v in self._last_load.box)
                cv2.rectangle(pequeno, (x0, y0), (x1, y1), (60, 180, 255), 2)
            # La etiqueta con la confianza, no solo el contorno. Un contorno sin
            # nombre obliga a quien mira la demo a adivinar si el sistema sabe QUÉ ha
            # visto o solo que hay algo ahí; y la confianza al lado es lo que hace
            # creíble el umbral de 0,50 cuando alguien pregunte por él.
            x0, y0 = (int(v * escala) for v in self._last_load.box[:2])
            texto = f"caja {self._last_load.confidence:.2f}"
            (ancho_t, alto_t), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            y0 = max(y0, alto_t + 8)
            cv2.rectangle(pequeno, (x0, y0 - alto_t - 7), (x0 + ancho_t + 8, y0 - 1), (60, 180, 255), -1)
            cv2.putText(pequeno, texto, (x0 + 4, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (20, 20, 20), 1, cv2.LINE_AA)

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
            px_cm = self._escala(keypoints)
        carga = build_load(
            cajas[indice],
            float(confianzas[indice]),
            px_cm,
            tuple(self.config.load_catalog) or None,
            poligonos[indice] if indice < len(poligonos) else None,
        )
        self._last_load = self._ancho_estable(carga, px_cm)

    def _ancho_estable(self, carga, px_cm: float | None):
        """Sustituye la anchura del fotograma por la mayor vista hace poco.

        La camara mide la proyeccion de la caja, no la caja: girada de canto se
        proyecta menos de la mitad. Como la proyeccion mas ancha es la anchura real,
        de la ventana reciente se toma el percentil 90 — el maximo dejaria que una
        sola deteccion pasada de grande mandara durante seis segundos.

        Y al cambiar la anchura hay que volver a emparejar con el catalogo, porque el
        peso que se aplica al levantamiento sale de ahi. Hacerlo a medias —anchura
        nueva, catalogo viejo— daria un peso que no corresponde a ninguna medida.
        """
        if carga.width_cm is None or px_cm is None:
            return carga
        self._anchos.append(carga.width_cm)
        if len(self._anchos) > ANCHO_VENTANA:
            self._anchos.pop(0)
        estable = round(float(np.percentile(self._anchos, 90)), 1)
        if estable <= carga.width_cm:
            return carga
        from ..load.carga import DetectedLoad, match_catalog

        catalogo = tuple(self.config.load_catalog) or None
        return DetectedLoad(
            box=carga.box,
            center=carga.center,
            confidence=carga.confidence,
            width_cm=estable,
            height_cm=carga.height_cm,
            matched=match_catalog(estable, catalogo),
            polygon=carga.polygon,
        )

    def _escala(self, keypoints: np.ndarray) -> float:
        """Píxeles por centímetro, tomados de lo más erguido que se le haya visto hace poco.

        La escala sale de medir de la coronilla al tobillo, y ese tramo **se encoge al
        agacharse**: la persona mide lo mismo y su proyección no. Medido sobre una
        grabación real (2026-10-02, 67 fotogramas), va de 2,16 a 4,41 px/cm en el mismo
        vídeo, así que una caja medida agachado sale 1,5× más grande — y el momento en
        que alguien levanta una caja es justo el momento en que está agachado.

        Por eso se guarda el máximo de una VENTANA de unos segundos y no de la sesión
        entera: dentro de unos segundos la persona no ha cambiado de distancia a la
        cámara, y a lo largo de un turno sí — y entonces el máximo histórico sería la
        escala de cuando pasó más cerca, que es otro error con el mismo disfraz.
        """
        actual = _pixels_per_cm(keypoints, self.config.worker_height_cm)
        self._escalas.append(actual)
        if len(self._escalas) > ESCALA_VENTANA:
            self._escalas.pop(0)
        return max(self._escalas)

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

    def _load_payload(self, caja: np.ndarray) -> dict | None:
        if self._last_load is None:
            return None
        c = self._last_load
        return {
            "canvas": self._canvas_load(caja),
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
            # Los fotogramas tolerados se reincorporan a la racha con el ultimo valor
            # conocido: su riesgo no fue menor, fue mal medido, y descontarlos de la
            # duracion haria que un evento de 1,2 s se publicara como de 1,0.
            if self._streak_gap and self._streak:
                self._streak.extend([self._streak[-1]] * self._streak_gap)
                self._streak_tasks.extend([self._streak_tasks[-1]] * self._streak_gap)
                self._streak_loads.extend([self._streak_loads[-1]] * self._streak_gap)
                if self._streak_poses:
                    self._streak_poses.extend([self._streak_poses[-1]] * self._streak_gap)
            self._streak_gap = 0
            self._streak.append((reba, peor))
            if keypoints is not None:
                self._streak_poses.append(keypoints)
            self._streak_tasks.append(self._last_task)
            self._streak_loads.append(
                np.array(self._last_load.center) if self._last_load else None
            )
            return
        if self._streak:
            # Un fotograma por debajo no cierra el tramo: se espera a ver si vuelve.
            if self._streak_gap < EVENT_GAP_FRAMES:
                self._streak_gap += 1
                return
            self._streak_gap = 0
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
            self._streak_gap = 0
            self._streak_poses = []
            self._streak_tasks = []
            self._streak_loads = []

    def close(self) -> None:
        self.capture.release()
