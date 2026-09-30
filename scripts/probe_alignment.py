"""¿Qué fotograma del esqueleto le corresponde a cada etiqueta de UW-IOM?

UW-IOM trae tres longitudes distintas por sujeto —esqueletos, fotogramas de vídeo
y etiquetas— y ninguna documentación dice cómo se alinean. Elegir mal desplaza el
dataset entero y los números salen coherentes y falsos, así que aquí no se supone:
se mide.

Las etiquetas viven en `VideoLabels`, o sea que etiquetan fotogramas de VÍDEO. El
esqueleto va en su propio reloj (`bodytimelogger`) y el vídeo en el suyo
(`videotimelogger`), así que la cadena correcta es
etiqueta → fotograma de vídeo → tiempo → fotograma de esqueleto más cercano.
Lo que falta por determinar es el primer eslabón: a qué fotogramas de vídeo
corresponden las etiquetas, dado que son menos que los fotogramas.

EL CRITERIO, escrito antes de ver los resultados: la etiqueta `bend` significa que
la persona está doblada, así que en los fotogramas etiquetados `bend` el tronco
tiene que estar más inclinado respecto a la vertical que en los `stand`. Gana la
hipótesis que maximice esa separación.

Y una advertencia aprendida en la primera versión de esta sonda: comparar el mejor
desplazamiento con el SEGUNDO mejor no dice nada, porque el segundo mejor es
siempre su vecino inmediato y los vecinos empatan por construcción. Lo que hay
que comparar son las hipótesis interpretables entre sí.

    .\\.venv\\Scripts\\python.exe scripts\\probe_alignment.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

BASE = Path(__file__).resolve().parents[1] / "data/raw/uw-iom/xwzzkxtf9s-2/UW IOM Dataset"

# Orden del SDK del Kinect v2. Se comprueba contra los datos, no se da por bueno.
SPINE_BASE, SPINE_SHOULDER = 0, 20


def trunk_inclination(joints3d: np.ndarray) -> np.ndarray:
    """Grados del tronco respecto a la vertical. 0 = erguido, 90 = horizontal."""
    trunk = joints3d[:, SPINE_SHOULDER, :] - joints3d[:, SPINE_BASE, :]
    norms = np.linalg.norm(trunk, axis=1)
    cos = np.divide(trunk[:, 1], norms, out=np.zeros(len(trunk)), where=norms > 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def load_subject(index: int):
    with h5py.File(BASE / "JointPositions" / f"{index}.mat", "r") as f:
        joints = np.array(f["bodylogger3D"])
        t_body = np.array(f["bodytimelogger"]).ravel()
        t_video = np.array(f["videotimelogger"]).ravel()
    labels = [
        line.strip()
        for line in (BASE / "VideoLabels" / f"{index:02d}.txt")
        .read_text(encoding="utf-8", errors="replace")
        .split("\n")
        if line.strip()
    ]
    return joints, t_body, t_video, labels


def separation(inclination: np.ndarray, body_idx: np.ndarray, labels: list[str]) -> float:
    """Cuántos grados más inclinado está el tronco en `bend` que en `stand`."""
    is_bend = np.array(["_bend_" in f"_{lab}_" for lab in labels])
    is_stand = np.array(["_stand_" in f"_{lab}_" for lab in labels])
    if is_bend.sum() < 10 or is_stand.sum() < 10:
        return float("nan")
    values = inclination[body_idx]
    return float(values[is_bend].mean() - values[is_stand].mean())


def map_by_time(t_body: np.ndarray, t_video: np.ndarray, video_idx: np.ndarray) -> np.ndarray:
    """Para cada fotograma de vídeo, el fotograma de esqueleto más cercano en tiempo."""
    return np.abs(t_body[None, :] - t_video[video_idx][:, None]).argmin(axis=1)


def main() -> None:
    joints, _, _, _ = load_subject(1)
    inc = trunk_inclination(joints)
    print("Comprobación previa del orden de articulaciones (sujeto 1):")
    print(
        f"  inclinación del tronco  p5 {np.percentile(inc, 5):.1f}°  "
        f"p50 {np.percentile(inc, 50):.1f}°  p95 {np.percentile(inc, 95):.1f}°"
    )
    assert np.percentile(inc, 50) < 30, "de pie la mediana debería ser baja: el orden no cuadra"
    print("  de pie sale erguido y doblado sale doblado: el orden del Kinect v2 es el bueno.\n")

    print("Separación bend-stand en grados. Más alto = etiquetas mejor alineadas.")
    print("suj | etiq al PRINCIPIO  al FINAL | por TIEMPO, principio  por TIEMPO, final")
    filas = []
    for i in range(1, 21):
        joints, t_body, t_video, labels = load_subject(i)
        inc = trunk_inclination(joints)
        n_body, n_video, n_lab = len(joints), len(t_video), len(labels)

        # Hipótesis ingenuas: la etiqueta k es el fotograma de ESQUELETO k (o k + hueco).
        directo_ini = separation(inc, np.arange(n_lab), labels)
        hueco_cuerpo = n_body - n_lab
        directo_fin = separation(inc, np.arange(hueco_cuerpo, hueco_cuerpo + n_lab), labels)

        # Hipótesis buenas: la etiqueta k es un fotograma de VÍDEO, y de ahí al
        # esqueleto por marca de tiempo.
        tiempo_ini = separation(inc, map_by_time(t_body, t_video, np.arange(n_lab)), labels)
        hueco_video = n_video - n_lab
        tiempo_fin = separation(
            inc, map_by_time(t_body, t_video, np.arange(hueco_video, hueco_video + n_lab)), labels
        )

        filas.append((directo_ini, directo_fin, tiempo_ini, tiempo_fin))
        print(
            f"{i:3d} | {directo_ini:16.2f} {directo_fin:9.2f} | "
            f"{tiempo_ini:21.2f} {tiempo_fin:17.2f}"
        )

    arr = np.array(filas)
    nombres = ["directo, principio", "directo, final", "por tiempo, principio", "por tiempo, final"]
    print("\nmediana sobre los 20 sujetos:")
    for nombre, col in zip(nombres, arr.T):
        print(f"  {nombre:24s} {np.nanmedian(col):6.2f}°")

    ganadora = int(np.nanargmax([np.nanmedian(c) for c in arr.T]))
    print(f"\nGana: {nombres[ganadora]}")
    por_sujeto = arr.argmax(axis=1)
    print(
        f"  y gana en {int((por_sujeto == ganadora).sum())} de 20 sujetos por separado, "
        "que es lo que dice si la conclusión es del conjunto o de la mediana."
    )


if __name__ == "__main__":
    main()
