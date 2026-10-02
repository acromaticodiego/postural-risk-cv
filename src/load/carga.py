"""La carga que se levanta: detectarla, saber si está en las manos, y cuánto pesa.

POR QUÉ ESTO NO ES «DETECTAR CAJAS PORQUE QUEDA BIEN». Detectar objetos es lo que
ya demuestran otros proyectos del portafolio; lo nuevo aquí es **fusionar persona y
carga para medir algo que ninguna de las dos por separado puede**:

1. **El peso deja de ser una constante del puesto.** Hoy ALMACEN-1 tiene 18 kg
   fijos, y por una línea real pasan cajas distintas. Con la carga medida, el
   cliente declara un catálogo —«caja grande, 45 cm, 18 kg»— y el sistema aplica el
   peso que corresponde a CADA levantamiento.
2. **NIOSH se vuelve fiel a la norma.** La ecuación mide la distancia horizontal
   desde el punto medio entre los tobillos hasta **el centro de la carga**. Sin ver
   la carga hay que usar las muñecas, que es una aproximación razonable y no lo que
   la norma pide.
3. **Confirma que hay un levantamiento**, por un camino independiente del modelo de
   tareas. Ese modelo ya sabemos que se equivoca fuera de su ángulo de cámara; ver
   una caja en las manos es otra evidencia, y dos evidencias que se contradicen son
   una señal, no un problema.

LA SEPARACIÓN QUE HACE ESTO COMPROBABLE: la inferencia (un modelo de segmentación)
vive en `LoadDetector`, y toda la geometría y las decisiones —si la caja está en
las manos, cuánto mide, qué peso le toca— son funciones puras que se prueban sin
GPU y sin modelo. Ahí es donde están los errores que importan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pose.schema import JOINT

# Cuánto puede alejarse el centro de una caja de las manos y seguir siendo «la carga
# que esa persona levanta», medido en alturas de cuerpo. Es CRITERIO, no norma: con
# el brazo extendido las manos y el centro de una caja mediana quedan a un tercio de
# altura de cuerpo, y por encima de eso ya suele ser una caja del fondo.
MAX_HAND_DISTANCE_BODIES = 0.35

# Confianza mínima del detector. Una caja dudosa que entra cambia el peso aplicado a
# un levantamiento y con él el índice de riesgo, así que más vale no verla.
#
# CALIBRADO el 2026-10-02 contra el modelo de la planta, con el criterio que el ADR
# 0001 ya tenía escrito —el sistema se inclina a no dar falsas alarmas—: **el umbral
# más bajo que no deja pasar ni un falso positivo**. El conjunto negativo es un vídeo
# del mismo sitio y la misma persona sosteniendo OTRO objeto, donde cualquier caja
# asociada a las manos es un error por construcción.
#
#   umbral   aciertos (67 con caja)   falsos (97 sin caja)
#    0,25           57                      10
#    0,45           52                       3
#    0,50           50                       0     <- elegido
#    0,60           46                       0
#    0,80           25                       0
#
# Subir por encima de 0,50 cuesta detecciones y no compra nada, porque los falsos ya
# son cero. Y con n=1 vídeo por lado, un objeto y una habitación, esto calibra ESTE
# despliegue; no es una tasa publicable.
MIN_CONFIDENCE = 0.50


@dataclass(frozen=True)
class LoadCatalogEntry:
    """Un tipo de carga que el cliente declara para un puesto."""

    name: str
    width_cm: float
    weight_kg: float


@dataclass(frozen=True)
class DetectedLoad:
    """Una carga detectada y asociada a la persona."""

    box: tuple[float, float, float, float]
    center: tuple[float, float]
    confidence: float
    width_cm: float | None
    height_cm: float | None
    matched: LoadCatalogEntry | None
    """La entrada del catálogo que mejor cuadra por tamaño. None si no hay catálogo
    o si ninguna se acerca lo suficiente."""
    polygon: list | None = None
    """El contorno para dibujar, en coordenadas de imagen. Solo si el modelo es de
    segmentación."""

    @property
    def weight_kg(self) -> float | None:
        return self.matched.weight_kg if self.matched else None


# --- geometría y decisiones: todo esto se prueba sin modelo -------------------


def hands_center(keypoints: np.ndarray) -> np.ndarray:
    """El punto medio entre las dos muñecas, que es por donde se agarra la carga."""
    return keypoints[[JOINT["left_wrist"], JOINT["right_wrist"]], :].mean(axis=0)


def body_height_px(keypoints: np.ndarray) -> float:
    validos = keypoints[(keypoints != 0).any(axis=1)]
    if not len(validos):
        return 1.0
    return max(float(validos[:, 1].max() - validos[:, 1].min()), 1.0)


def associate_to_person(
    boxes: np.ndarray, keypoints: np.ndarray, max_distance_bodies: float = MAX_HAND_DISTANCE_BODIES
) -> int | None:
    """De todas las cargas detectadas, cuál está en las manos. None si ninguna.

    Se mide del centro de la caja al punto medio de las muñecas, normalizado por la
    altura del cuerpo para que no dependa de la distancia a la cámara. Si hay varias
    candidatas dentro del radio, gana la más cercana: en un almacén hay cajas por
    todas partes y la que importa es la que se está levantando.

    Devolver None cuando no hay ninguna cerca es la mitad del trabajo. Asociar a la
    fuerza la caja más próxima haría que el sistema aplicara un peso a alguien que
    solo está caminando, y ese peso entra en el cálculo de riesgo.
    """
    if boxes is None or not len(boxes):
        return None
    manos = hands_center(keypoints)
    if (manos == 0).all():
        return None
    altura = body_height_px(keypoints)

    centros = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    distancias = np.linalg.norm(centros - manos, axis=1) / altura
    mejor = int(distancias.argmin())
    return mejor if distancias[mejor] <= max_distance_bodies else None


def load_size_cm(box, px_per_cm: float) -> tuple[float, float]:
    """El tamaño de la carga en centímetros, con la escala del cuerpo."""
    ancho = abs(float(box[2]) - float(box[0])) / max(px_per_cm, 1e-9)
    alto = abs(float(box[3]) - float(box[1])) / max(px_per_cm, 1e-9)
    return round(ancho, 1), round(alto, 1)


def match_catalog(
    width_cm: float, catalog: tuple[LoadCatalogEntry, ...] | None, tolerance: float = 0.35
) -> LoadCatalogEntry | None:
    """Qué carga del catálogo cuadra con lo medido, por anchura.

    `tolerance` es la desviación relativa admitida. Fuera de ella se devuelve None
    en vez de la más parecida: si el cliente declaró cajas de 30 y 45 cm y la cámara
    mide 80, lo honesto es decir que eso no está en el catálogo, no asignarle el
    peso de la más grande y seguir como si nada.
    """
    if not catalog:
        return None
    candidatas = [
        (abs(entrada.width_cm - width_cm) / max(entrada.width_cm, 1e-9), entrada)
        for entrada in catalog
    ]
    error, mejor = min(candidatas, key=lambda par: par[0])
    return mejor if error <= tolerance else None


def build_load(
    box,
    confidence: float,
    px_per_cm: float | None,
    catalog: tuple[LoadCatalogEntry, ...] | None,
    polygon: list | None = None,
) -> DetectedLoad:
    """Arma la carga ya medida y emparejada con el catálogo."""
    ancho = alto = None
    emparejada = None
    if px_per_cm:
        ancho, alto = load_size_cm(box, px_per_cm)
        emparejada = match_catalog(ancho, catalog)
    return DetectedLoad(
        box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
        center=((float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2),
        confidence=round(float(confidence), 3),
        width_cm=ancho,
        height_cm=alto,
        matched=emparejada,
        polygon=polygon,
    )


# --- la parte que necesita el modelo, deliberadamente fina -------------------


class LoadDetector:
    """Envuelve el modelo de segmentación. Todo lo demás vive fuera a propósito."""

    def __init__(self, weights: str, device: int | str = 0, min_confidence: float = MIN_CONFIDENCE):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.min_confidence = min_confidence

    def detect(self, frame: np.ndarray):
        """Devuelve (cajas, confianzas, polígonos) de las cargas por encima del corte."""
        resultado = self.model.predict(frame, device=self.device, verbose=False)[0]
        if resultado.boxes is None or not len(resultado.boxes):
            return np.empty((0, 4)), np.empty(0), []
        cajas = resultado.boxes.xyxy.cpu().numpy()
        confianzas = resultado.boxes.conf.cpu().numpy()
        quedan = confianzas >= self.min_confidence
        poligonos = []
        if resultado.masks is not None:
            poligonos = [
                np.round(p, 1).tolist()
                for p, ok in zip(resultado.masks.xy, quedan)
                if ok
            ]
        return cajas[quedan], confianzas[quedan], poligonos
