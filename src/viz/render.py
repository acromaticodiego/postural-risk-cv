"""El replay en esqueleto: la pieza que hace la demo.

Reproduce lo que ocurrió con el esqueleto, los ángulos y el puntaje REBA en vivo,
**sin imagen de la persona**. No es una limitación que haya que disculpar: es la
garantía de privacidad demostrada en pantalla. Cualquiera que vea el vídeo entiende
en dos segundos que el sistema no guarda caras, y a la vez ve exactamente qué
postura disparó el riesgo.

Se dibuja desde el `.npz` de keypoints, así que esto se puede generar meses después
del turno sin haber conservado un solo fotograma.

    .\\.venv\\Scripts\\python.exe -m src.viz.render 1 --out artifacts/replay-01.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ..baseline.reba import action_level, reba_from_keypoints
from ..pose.schema import JOINT
from ..product.workstation import WorkstationConfig, summarize_exposure

WIDTH, HEIGHT = 1280, 720
PANEL_W = 420
STAGE_W = WIDTH - PANEL_W

FONDO = (22, 18, 15)
PANEL = (32, 27, 23)
TEXTO = (232, 230, 227)
SUAVE = (150, 146, 141)
REJILLA = (48, 42, 37)

# Colores por nivel de riesgo, en BGR. Del verde al rojo pasando por ámbar, que es
# el código que un ingeniero de seguridad ya lee sin explicación.
COLOR_NIVEL = {
    "despreciable": (120, 190, 110),
    "bajo": (130, 200, 150),
    "medio": (70, 190, 235),
    "alto": (60, 120, 245),
    "muy alto": (75, 70, 235),
    "sin persona": (90, 85, 80),
}

# El esqueleto COCO: qué articulaciones se unen con qué.
HUESOS = (
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
)

ANGULOS_MOSTRADOS = (
    ("trunk_flexion", "Tronco", "trunk"),
    ("neck_flexion", "Cuello", "neck"),
    ("upper_arm_elevation", "Brazo", "upper_arm"),
    ("lower_arm_flexion", "Antebrazo", "lower_arm"),
    ("knee_flexion", "Rodilla", "legs"),
)


def _fit_transform(keypoints: np.ndarray, present: np.ndarray) -> tuple[float, np.ndarray]:
    """Una sola transformación para toda la secuencia, no una por fotograma.

    Reencuadrar en cada fotograma haría que el esqueleto pareciera quieto mientras
    el mundo se mueve, que es justo lo contrario de lo que hay que mostrar: la
    persona se agacha y tiene que VERSE agacharse.
    """
    validos = keypoints[present]
    if not len(validos):
        raise ValueError("la secuencia no tiene ningún fotograma con persona")
    puntos = validos.reshape(-1, 2)
    puntos = puntos[(puntos != 0).any(axis=1)]
    minimo, maximo = puntos.min(axis=0), puntos.max(axis=0)
    tamano = np.maximum(maximo - minimo, 1.0)
    margen = 0.12
    escala = min(STAGE_W * (1 - 2 * margen) / tamano[0], HEIGHT * (1 - 2 * margen) / tamano[1])
    centro_origen = (minimo + maximo) / 2
    centro_destino = np.array([STAGE_W / 2, HEIGHT / 2])
    return escala, centro_destino - centro_origen * escala


def _draw_text(img, texto, pos, escala=0.5, color=TEXTO, grosor=1) -> None:
    cv2.putText(img, texto, pos, cv2.FONT_HERSHEY_DUPLEX, escala, color, grosor, cv2.LINE_AA)


def _draw_skeleton(lienzo, puntos, color) -> None:
    for a, b in HUESOS:
        pa, pb = puntos[JOINT[a]], puntos[JOINT[b]]
        if (pa == 0).all() or (pb == 0).all():
            continue
        cv2.line(lienzo, tuple(pa.astype(int)), tuple(pb.astype(int)), color, 4, cv2.LINE_AA)
    for nombre in JOINT:
        p = puntos[JOINT[nombre]]
        if (p == 0).all():
            continue
        cv2.circle(lienzo, tuple(p.astype(int)), 5, color, -1, cv2.LINE_AA)
        cv2.circle(lienzo, tuple(p.astype(int)), 5, FONDO, 1, cv2.LINE_AA)


def _draw_panel(lienzo, datos, indice, config, acumulado, hz) -> None:
    x0 = STAGE_W
    cv2.rectangle(lienzo, (x0, 0), (WIDTH, HEIGHT), PANEL, -1)
    cv2.line(lienzo, (x0, 0), (x0, HEIGHT), REJILLA, 1)

    reba = int(datos["reba"][indice])
    nivel = action_level(reba) if reba > 0 else "sin persona"
    color = COLOR_NIVEL[nivel]

    _draw_text(lienzo, config.name.upper(), (x0 + 28, 46), 0.52, SUAVE)
    _draw_text(lienzo, "RIESGO POSTURAL  ·  REBA", (x0 + 28, 72), 0.42, SUAVE)

    # El puntaje, grande, con su nivel. Es lo primero que se mira.
    cv2.rectangle(lienzo, (x0 + 28, 92), (x0 + 392, 188), color, 2, cv2.LINE_AA)
    _draw_text(lienzo, f"{reba if reba else '--'}", (x0 + 48, 168), 2.4, color, 3)
    _draw_text(lienzo, nivel.upper(), (x0 + 168, 140), 0.66, color, 1)
    _draw_text(lienzo, f"umbral de accion: {config.risk_threshold}", (x0 + 168, 168), 0.4, SUAVE)

    # Los componentes: el valor del ángulo y el puntaje que le da la norma. Esto es
    # lo que convierte "riesgo alto" en "el tronco, y por esto".
    y = 232
    _draw_text(lienzo, "COMPONENTES", (x0 + 28, y), 0.42, SUAVE)
    y += 26
    for clave, etiqueta, componente in ANGULOS_MOSTRADOS:
        grados = float(datos[clave][indice])
        puntaje = int(datos[componente][indice])
        _draw_text(lienzo, etiqueta, (x0 + 28, y + 12), 0.46, TEXTO)
        _draw_text(lienzo, f"{grados:5.0f}°", (x0 + 148, y + 12), 0.46, SUAVE)
        # Barra proporcional al puntaje del componente, sobre su máximo de la norma.
        maximo = 4 if componente in ("trunk", "upper_arm", "legs") else 2
        ancho = int(150 * min(puntaje / maximo, 1.0))
        cv2.rectangle(lienzo, (x0 + 218, y), (x0 + 368, y + 14), REJILLA, -1)
        if ancho:
            tono = COLOR_NIVEL["medio"] if puntaje / maximo < 0.75 else COLOR_NIVEL["alto"]
            cv2.rectangle(lienzo, (x0 + 218, y), (x0 + 218 + ancho, y + 14), tono, -1)
        _draw_text(lienzo, str(puntaje), (x0 + 376, y + 12), 0.42, SUAVE)
        y += 30

    # Exposición acumulada hasta este fotograma, por nivel. Es la cifra del informe.
    y += 18
    _draw_text(lienzo, "EXPOSICION ACUMULADA", (x0 + 28, y), 0.42, SUAVE)
    y += 26
    for nombre in ("medio", "alto", "muy alto"):
        segundos = acumulado.get(nombre, 0.0)
        _draw_text(lienzo, nombre, (x0 + 28, y + 12), 0.46, COLOR_NIVEL[nombre])
        _draw_text(lienzo, f"{segundos:6.1f} s", (x0 + 300, y + 12), 0.46, TEXTO)
        y += 28

    y += 16
    transcurrido = indice / hz
    _draw_text(lienzo, f"t = {transcurrido:5.1f} s", (x0 + 28, y + 12), 0.46, SUAVE)
    if not config.is_configured:
        y += 30
        _draw_text(lienzo, "puesto sin configurar:", (x0 + 28, y + 12), 0.4, COLOR_NIVEL["medio"])
        _draw_text(lienzo, "el riesgo real es MAYOR", (x0 + 28, y + 30), 0.4, COLOR_NIVEL["medio"])


def render_replay(
    keypoints: np.ndarray,
    present: np.ndarray,
    hz: float,
    config: WorkstationConfig,
    out_path: str | Path,
    max_frames: int | None = None,
) -> Path:
    datos = reba_from_keypoints(keypoints, present=present, assumptions=config.assumptions())
    escala, desplazamiento = _fit_transform(keypoints, present)
    n = len(keypoints) if max_frames is None else min(max_frames, len(keypoints))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    escritor = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), hz, (WIDTH, HEIGHT)
    )

    acumulado: dict[str, float] = {}
    for i in range(n):
        reba = int(datos["reba"][i])
        if reba > 0:
            nivel = action_level(reba)
            acumulado[nivel] = acumulado.get(nivel, 0.0) + 1.0 / hz

        lienzo = np.full((HEIGHT, WIDTH, 3), FONDO, dtype=np.uint8)
        for gx in range(0, STAGE_W, 80):
            cv2.line(lienzo, (gx, 0), (gx, HEIGHT), REJILLA, 1)
        for gy in range(0, HEIGHT, 80):
            cv2.line(lienzo, (0, gy), (STAGE_W, gy), REJILLA, 1)

        nivel_actual = action_level(reba) if reba > 0 else "sin persona"
        if present[i]:
            _draw_skeleton(lienzo, keypoints[i] * escala + desplazamiento, COLOR_NIVEL[nivel_actual])
        else:
            _draw_text(lienzo, "sin persona en cuadro", (40, HEIGHT - 40), 0.5, SUAVE)

        _draw_text(lienzo, "REPLAY EN ESQUELETO", (32, 44), 0.54, TEXTO)
        _draw_text(
            lienzo,
            "reconstruido de articulaciones: no se guardo ninguna imagen",
            (32, 70),
            0.42,
            SUAVE,
        )
        _draw_panel(lienzo, datos, i, config, acumulado, hz)
        escritor.write(lienzo)

    escritor.release()
    return out_path


def _cli() -> None:
    from ..datasets.uwiom import load_subject
    from ..datasets.windows import resample

    parser = argparse.ArgumentParser(description="Replay en esqueleto de un sujeto.")
    parser.add_argument("subject", type=int)
    parser.add_argument("--out", default=None)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--load-kg", type=float, default=None)
    parser.add_argument("--coupling", default=None)
    args = parser.parse_args()

    sujeto = load_subject(args.subject)
    elegidos, _, presente = resample(sujeto, args.hz)
    config = WorkstationConfig(
        id=f"UWIOM-{args.subject:02d}",
        name=f"Puesto de prueba {args.subject:02d}",
        load_kg=args.load_kg,
        coupling=args.coupling,
    )
    destino = args.out or f"artifacts/replay-{args.subject:02d}.mp4"
    ruta = render_replay(
        sujeto.skeleton.keypoints[elegidos],
        presente,
        args.hz,
        config,
        destino,
        max_frames=args.max_frames,
    )

    datos = reba_from_keypoints(
        sujeto.skeleton.keypoints[elegidos], present=presente, assumptions=config.assumptions()
    )
    resumen = summarize_exposure(datos, args.hz, config, present=presente)
    print(f"{ruta}  ({ruta.stat().st_size / 1e6:.1f} MB)")
    print(f"  medido {resumen.measured_seconds:.0f} s | en riesgo {resumen.seconds_at_risk:.1f} s")
    print(f"  eventos: {len(resumen.events)}")
    for e in resumen.events[:5]:
        print(
            f"    t={e.start_seconds:6.1f}s  {e.duration_seconds:4.1f}s  "
            f"pico {e.peak_reba} ({e.peak_level})  por {e.dominant_component}"
        )


if __name__ == "__main__":
    _cli()
