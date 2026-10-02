"""Pre-etiquetar la carga con SAM guiado por el esqueleto, y decidir qué máscara vale.

POR QUÉ ESTO EXISTE. El detector de carga viene entrenado con `package-seg`: cajas
de cartón de almacén, en estanterías y cintas transportadoras. En las manos de una
persona no se sabe si funciona (medición falsa nº 5). La salida de producto no es
sustituir ese modelo, es **afinarlo con las cargas del cliente** — que es justo lo
que haría un sistema instalable: viene de fábrica y se calibra en cada planta.

Para afinarlo hacen falta máscaras, y dibujarlas a mano es lo que mata este tipo de
proyecto. Aquí no se dibujan: **el esqueleto ya dice dónde está la carga.** Si una
persona sostiene algo, está entre sus manos; y lo que SAM no debe segmentar es la
propia persona, cuyas articulaciones también conocemos. De ahí salen los puntos de
aviso: uno POSITIVO entre las muñecas y varios NEGATIVOS repartidos por el cuerpo.

LA SEPARACIÓN QUE HACE ESTO COMPROBABLE, igual que en `carga.py`: SAM vive fuera de
este módulo. Aquí están el armado de los puntos, los filtros de plausibilidad y la
conversión a etiqueta de YOLO, que es donde están los errores que importan y se
prueban sin GPU y sin modelo.

Y UNA ADVERTENCIA QUE NO ES OPCIONAL: un pre-etiquetado no es una etiqueta. Lo que
sale de aquí es una propuesta que una persona valida; los filtros existen para que
esa validación sea mirar y descartar, no corregir. Entrenar con lo que salga de
aquí sin que nadie lo haya mirado es exactamente la forma de obtener un número
limpio sobre material falso.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pose.schema import JOINT
from .carga import MAX_HAND_DISTANCE_BODIES, body_height_px, hands_center

# Articulaciones que marcan «esto NO es la carga». Son el torso, la cara y las
# piernas: lo que SAM confundiría con el objeto si solo se le diera el punto de las
# manos, porque una caja pegada al pecho y el pecho son una sola mancha de píxeles.
NEGATIVE_JOINTS: tuple[str, ...] = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
)

# Articulaciones que una carga NO puede estar tapando, y que por eso sirven para
# rechazar una máscara. La lista es corta a propósito: la primera versión incluía las
# caderas y rechazaba el caso bueno, porque una caja sujetada contra el vientre SÍ
# las tapa en la imagen. Lo que no tapa nunca es la cara de quien la lleva.
FORBIDDEN_JOINTS: tuple[str, ...] = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")

# Qué parte de la altura de la persona puede ocupar su carga. Es el filtro que separa
# «una caja» de «SAM ha segmentado a la persona»: una caja de 50 cm en las manos de
# alguien de 1,70 m ocupa menos de un tercio de su altura en la imagen, y la silueta
# de la persona ocupa el 100% por definición. Es CRITERIO, no norma.
MAX_LOAD_HEIGHT_FRAC = 0.60

# Confianza mínima de una articulación para usarla como punto de aviso. Un punto
# negativo mal puesto es peor que no ponerlo: si cae encima de la caja, le estamos
# diciendo a SAM que la caja no es la caja.
MIN_JOINT_SCORE = 0.50

# Las dos muñecas tienen que verse para situar la carga. Es el requisito más duro y
# descarta fotogramas, a propósito: sin saber dónde están las manos, el punto
# positivo se pone en un sitio inventado y la máscara sale de cualquier cosa.
MIN_WRIST_SCORE = 0.50

# Cuánto puede ocupar la carga en la imagen. Por debajo del mínimo es ruido —una
# mano, un reflejo— y por encima del máximo SAM se ha llevado media escena, que es
# su fallo típico cuando el objeto y el fondo tienen el mismo color.
MIN_MASK_AREA_FRAC = 0.004
MAX_MASK_AREA_FRAC = 0.35

# La máscara tiene que ser UNA cosa. Si la componente conexa mayor no llega a este
# porcentaje del área, lo que hay son trozos sueltos repartidos por la imagen, y eso
# no es una caja: es SAM habiendo seguido una textura.
MIN_LARGEST_COMPONENT_FRAC = 0.70

# Un polígono de YOLO-seg con menos de 3 vértices no es un polígono.
MIN_POLYGON_POINTS = 3


@dataclass(frozen=True)
class Prompt:
    """Los puntos con los que se le pregunta a SAM dónde está la carga."""

    points: list[list[float]]
    labels: list[int]

    @property
    def positives(self) -> int:
        return sum(self.labels)

    @property
    def negatives(self) -> int:
        return len(self.labels) - self.positives


@dataclass(frozen=True)
class Verdict:
    """Si una máscara propuesta se acepta, y si no, por qué no.

    El motivo se guarda siempre, también en las aceptadas. Un pre-etiquetado del que
    solo se sabe el sí o el no no se puede depurar: cuando el 60% salga rechazado hay
    que poder contar POR QUÉ sin volver a correr nada.
    """

    ok: bool
    reason: str


def build_prompt(
    keypoints: np.ndarray,
    scores: np.ndarray,
    min_joint_score: float = MIN_JOINT_SCORE,
    min_wrist_score: float = MIN_WRIST_SCORE,
) -> Prompt | None:
    """Los puntos de aviso para SAM, sacados del esqueleto. None si no se puede.

    El positivo va en el punto medio entre las muñecas, que es por donde se agarra
    cualquier carga, y los negativos en las articulaciones del cuerpo que se vean con
    confianza suficiente.

    Devolver None cuando las muñecas no se ven es la mitad del trabajo, y es
    deliberado que descarte fotogramas: la alternativa —poner el punto donde caiga—
    produce máscaras plausibles de cosas que no son la carga, y esas son las que se
    cuelan en la validación humana porque parecen correctas.
    """
    if keypoints.shape[0] != len(JOINT) or scores.shape[0] != len(JOINT):
        raise ValueError(f"se esperaban {len(JOINT)} articulaciones")

    izq, der = JOINT["left_wrist"], JOINT["right_wrist"]
    if scores[izq] < min_wrist_score or scores[der] < min_wrist_score:
        return None

    centro = hands_center(keypoints)
    puntos = [[float(centro[0]), float(centro[1])]]
    etiquetas = [1]

    for nombre in NEGATIVE_JOINTS:
        j = JOINT[nombre]
        if scores[j] >= min_joint_score:
            puntos.append([float(keypoints[j, 0]), float(keypoints[j, 1])])
            etiquetas.append(0)

    # Sin un solo negativo, SAM se lleva a la persona entera siempre que la caja esté
    # pegada al cuerpo, que es el caso normal al levantar. Preferimos no proponer
    # nada antes que proponer el contorno de alguien.
    if sum(1 for e in etiquetas if e == 0) == 0:
        return None

    return Prompt(points=puntos, labels=etiquetas)


def largest_component_fraction(mask: np.ndarray) -> float:
    """Qué parte de la máscara está en su trozo más grande. 1.0 si es una sola pieza."""
    import cv2

    binaria = (mask > 0).astype(np.uint8)
    total = int(binaria.sum())
    if total == 0:
        return 0.0
    n, etiquetas = cv2.connectedComponents(binaria)
    if n <= 1:
        return 0.0
    tamanos = [int((etiquetas == i).sum()) for i in range(1, n)]
    return max(tamanos) / total


def judge_mask(
    mask: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    min_joint_score: float = MIN_JOINT_SCORE,
    max_hand_distance_bodies: float = MAX_HAND_DISTANCE_BODIES,
) -> Verdict:
    """¿Esa máscara es una carga en las manos, o es otra cosa?

    Cinco filtros, y cada uno nace de un fallo concreto de SAM guiado por puntos:

      · **el tamaño**, porque cuando el objeto y el fondo comparten color SAM se
        lleva media imagen, y cuando el punto cae en un hueco se lleva cuatro
        píxeles;
      · **que sea una sola pieza**, porque SAM sigue texturas y devuelve manchas
        repartidas que suman un área razonable sin ser ningún objeto;
      · **que no tape la cara**, que es la única parte del cuerpo que quien carga
        algo no puede tener detrás de la carga;
      · **que no sea tan alta como la persona**, que es lo que de verdad separa una
        caja de una silueta. SAM respeta los puntos negativos casi siempre, y «casi»
        no es una garantía;
      · **que esté en las manos**, con el mismo criterio y el mismo radio que usa el
        sistema en producción para decidir si una caja es la que alguien levanta. Si
        aquí se usara otro, estaríamos etiquetando cargas que el sistema luego
        ignoraría.

    Lo que NO se comprueba, y es una corrección: que la máscara no toque las caderas
    ni el torso. La primera versión lo hacía, sobre las mismas articulaciones que se
    le pasan a SAM como negativas, y rechazaba el caso bueno — una caja sujetada
    contra el vientre tapa las caderas en la imagen, y eso es lo normal al levantar.
    """
    binaria = mask > 0
    area = float(binaria.mean())
    if area < MIN_MASK_AREA_FRAC:
        return Verdict(False, f"demasiado pequena ({area:.4f} del fotograma)")
    if area > MAX_MASK_AREA_FRAC:
        return Verdict(False, f"demasiado grande ({area:.3f} del fotograma)")

    pieza = largest_component_fraction(mask)
    if pieza < MIN_LARGEST_COMPONENT_FRAC:
        return Verdict(False, f"en trozos sueltos (la mayor es el {pieza:.0%})")

    alto, ancho = binaria.shape[:2]
    for nombre in FORBIDDEN_JOINTS:
        j = JOINT[nombre]
        if scores[j] < min_joint_score:
            continue
        x, y = int(round(keypoints[j, 0])), int(round(keypoints[j, 1]))
        if 0 <= x < ancho and 0 <= y < alto and binaria[y, x]:
            return Verdict(False, f"tapa la cara (cubre {nombre})")

    ys, xs = np.nonzero(binaria)
    altura_cuerpo = body_height_px(keypoints)
    altura_mascara = float(ys.max() - ys.min() + 1)
    proporcion = altura_mascara / altura_cuerpo
    if proporcion > MAX_LOAD_HEIGHT_FRAC:
        return Verdict(False, f"tan alta como la persona ({proporcion:.0%} del cuerpo)")

    centro = np.array([xs.mean(), ys.mean()])
    manos = hands_center(keypoints)
    distancia = float(np.linalg.norm(centro - manos) / altura_cuerpo)
    if distancia > max_hand_distance_bodies:
        return Verdict(False, f"lejos de las manos ({distancia:.2f} alturas de cuerpo)")

    return Verdict(
        True,
        f"area {area:.3f}, una pieza al {pieza:.0%}, "
        f"{proporcion:.0%} del cuerpo, a {distancia:.2f} de las manos",
    )


def mask_to_polygon(mask: np.ndarray, epsilon_frac: float = 0.004) -> np.ndarray | None:
    """El contorno de la máscara como polígono, en píxeles. None si no hay contorno.

    Se simplifica con Douglas-Peucker: un contorno en bruto trae un vértice por píxel
    de borde y una etiqueta de YOLO con dos mil vértices es ilegible para una persona
    que la tiene que validar, además de pesar diez veces más. `epsilon_frac` es la
    tolerancia como fracción del perímetro, así que la simplificación es la misma
    para una caja cerca y otra lejos.
    """
    import cv2

    binaria = (mask > 0).astype(np.uint8)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    mayor = max(contornos, key=cv2.contourArea)
    epsilon = epsilon_frac * cv2.arcLength(mayor, True)
    simplificado = cv2.approxPolyDP(mayor, epsilon, True).reshape(-1, 2)
    if len(simplificado) < MIN_POLYGON_POINTS:
        return None
    return simplificado.astype(np.float32)


def polygon_to_yolo_label(polygon: np.ndarray, width: int, height: int, cls: int = 0) -> str:
    """Una línea de etiqueta de YOLO-seg: clase y el polígono normalizado a 0..1.

    Se recorta a [0, 1] en vez de dejar pasar coordenadas fuera del fotograma. Un
    vértice en -0.01 no revienta el entrenamiento: lo acepta y desplaza la máscara en
    silencio, que es peor.
    """
    if width <= 0 or height <= 0:
        raise ValueError("el tamaño de la imagen tiene que ser positivo")
    normalizado = polygon.astype(np.float64).copy()
    normalizado[:, 0] /= width
    normalizado[:, 1] /= height
    normalizado = np.clip(normalizado, 0.0, 1.0)
    coordenadas = " ".join(f"{v:.6f}" for v in normalizado.reshape(-1))
    return f"{cls} {coordenadas}"


def pose_vector(keypoints: np.ndarray, box: np.ndarray) -> np.ndarray:
    """El esqueleto como un vector comparable entre fotogramas y entre distancias.

    Sirve para elegir fotogramas VARIADOS: dos posturas se parecen si este vector se
    parece. Se normaliza por la caja de la persona con la MISMA escala en los dos
    ejes —la razón está en `pose/schema.py`— porque si no, acercarse a la cámara
    contaría como cambiar de postura.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    escala = max(x1 - x0, y1 - y0, 1.0)
    centro = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    return ((keypoints[:, :2] - centro) / escala).reshape(-1)


def pick_varied(vectors: list[np.ndarray], times: list[float], n: int, min_gap_s: float) -> list[int]:
    """Qué fotogramas quedarse: los más distintos entre sí, no los más repartidos.

    Muestrear uno cada N segundos es lo cómodo y da un conjunto donde la mitad de los
    fotogramas son la misma persona de pie esperando. Esto hace lo contrario: empieza
    por el primero y va añadiendo, cada vez, el fotograma MÁS LEJANO en postura a todo
    lo ya elegido. Es muestreo del punto más lejano, y produce un conjunto donde cada
    imagen aporta algo que no estaba.

    `min_gap_s` sigue haciendo falta encima de eso: dos fotogramas separados por 40 ms
    pueden medir distinto por el ruido del detector sin ser posturas distintas, y un
    par casi idéntico repartido entre entrenamiento y validación infla el resultado.
    """
    if n <= 0 or not vectors:
        return []
    elegidos = [0]
    distancias = np.array([float(np.linalg.norm(v - vectors[0])) for v in vectors])

    while len(elegidos) < min(n, len(vectors)):
        candidato, mejor = -1, -1.0
        for i in range(len(vectors)):
            if i in elegidos:
                continue
            if any(abs(times[i] - times[j]) < min_gap_s for j in elegidos):
                continue
            if distancias[i] > mejor:
                candidato, mejor = i, float(distancias[i])
        if candidato < 0:
            break
        elegidos.append(candidato)
        nuevas = np.array([float(np.linalg.norm(v - vectors[candidato])) for v in vectors])
        distancias = np.minimum(distancias, nuevas)

    return sorted(elegidos)
