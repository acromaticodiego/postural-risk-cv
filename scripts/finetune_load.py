"""Afinar el detector de carga con las cargas del cliente, sin que olvide las de fábrica.

LA DECISIÓN DE PRODUCTO QUE GOBIERNA ESTE GUION: se AFINA el modelo base, no se
sustituye. Los vídeos de un cliente son una habitación, una luz y dos o tres cajas;
entrenar solo con eso aprende *esas* cajas y se cae con la siguiente. Afinar parte de
`package-seg` —2.197 imágenes de cajas de almacén— y le añade las del cliente, que es
exactamente el ciclo de vida de un producto instalable: el modelo viene de fábrica y
se calibra en cada planta. Lo que aquí es una limitación del material resulta ser la
arquitectura correcta.

LAS TRES COSAS QUE HACEN QUE EL NÚMERO SIGNIFIQUE ALGO, y sin las cuales esto sería
un entrenamiento con una métrica bonita:

1. **La partición es POR VÍDEO, no por fotograma.** Dos fotogramas del mismo vídeo
   separados por medio segundo son casi la misma imagen; repartidos entre
   entrenamiento y validación, el modelo aprueba por haberlos memorizado. Es la misma
   regla que la partición por sujeto del modelo de tareas, y por el mismo motivo.
2. **Se mide ANTES y DESPUÉS.** Un mAP a secas no dice si afinar sirvió de algo.
3. **Se mide también sobre `package-seg`**, que es el conjunto que el modelo ya sabía.
   Afinar con cien imágenes de una habitación puede subir el número del cliente y
   destrozar el de fábrica —olvido catastrófico—, y sin medirlo no se vería: el
   conjunto del cliente seguiría dando un número excelente.

Y SE NIEGA A ENTRENAR CON CAPTURAS DE PANTALLA. El material declarado como `pantalla`
lleva el esqueleto del panel PINTADO sobre la persona y sobre la carga, así que el
modelo aprendería a buscar líneas de colores y el número saldría bien porque la
validación lleva las mismas líneas. La garantía vive aquí, en el código, y no en un
aviso del documento: un aviso se lee una vez y se desobedece a la tercera sesión.

    .\\.venv\\Scripts\\python.exe scripts\\finetune_load.py --val-video carga_lateral
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RAIZ = Path(__file__).resolve().parents[1]
PESOS_BASE = RAIZ / "artifacts/modelo/carga.pt"
BASE_DATASET = Path("C:/Users/ASUS/Desktop/datasets/package-seg")

# Cuántas imágenes de fábrica se mezclan por cada imagen nueva. Con las 1.920 de
# `package-seg` enteras, las del cliente serían el 5% del lote y el afinado apenas se
# notaría; sin ninguna, el modelo olvida las cajas de almacén. 3 a 1 es el punto medio
# y es un parámetro, no una constante escondida: el artefacto guarda cuál se usó.
BASE_POR_NUEVO = 3


def _cargar(datos: Path) -> tuple[dict, list[dict]]:
    manifiesto = json.loads((datos / "manifiesto.json").read_text(encoding="utf-8"))
    prelabel = json.loads((datos / "prelabel.json").read_text(encoding="utf-8"))

    rechazadas: set[str] = set()
    ruta_rechazos = datos / "rechazadas.txt"
    if ruta_rechazos.exists():
        rechazadas = {
            linea.strip() for linea in ruta_rechazos.read_text(encoding="utf-8").splitlines()
            if linea.strip()
        }
        print(f"{len(rechazadas)} descartadas a mano en {ruta_rechazos.name}")
    else:
        print(f"AVISO: no hay {ruta_rechazos.name}. Nadie ha validado el pre-etiquetado.")

    por_archivo = {f["archivo"]: f for f in manifiesto["frames"]}
    buenas = [
        {**por_archivo[p["archivo"]], "motivo": p["motivo"]}
        for p in prelabel["propuestas"]
        if p["archivo"] not in rechazadas and p["archivo"] in por_archivo
    ]
    return manifiesto, buenas


def _construir(datos: Path, buenas: list[dict], val_video: str, base_por_nuevo: int) -> Path:
    """Arma el dataset de YOLO con la partición por vídeo y la mezcla de fábrica."""
    import random

    destino = datos / "dataset"
    if destino.exists():
        shutil.rmtree(destino)
    for parte in ("train", "val"):
        (destino / "images" / parte).mkdir(parents=True)
        (destino / "labels" / parte).mkdir(parents=True)

    reparto = {"train": 0, "val": 0}
    for f in buenas:
        parte = "val" if f["video"] == val_video else "train"
        nombre = Path(f["archivo"]).stem
        shutil.copy2(datos / "frames" / f["archivo"], destino / "images" / parte / f["archivo"])
        shutil.copy2(datos / "labels" / f"{nombre}.txt", destino / "labels" / parte / f"{nombre}.txt")
        reparto[parte] += 1

    # Las de fábrica van SOLO a entrenamiento. En validación estorbarían: lo que esa
    # mitad tiene que responder es si el modelo funciona con las cargas del cliente, y
    # `package-seg` se evalúa aparte con su propio conjunto de validación.
    base_imagenes = sorted((BASE_DATASET / "images/train").glob("*.jpg"))
    cuantas = min(len(base_imagenes), reparto["train"] * base_por_nuevo)
    random.Random(0).shuffle(base_imagenes)
    mezcladas = 0
    for imagen in base_imagenes[:cuantas]:
        etiqueta = BASE_DATASET / "labels/train" / f"{imagen.stem}.txt"
        if not etiqueta.exists():
            continue
        shutil.copy2(imagen, destino / "images/train" / f"base_{imagen.name}")
        shutil.copy2(etiqueta, destino / "labels/train" / f"base_{imagen.stem}.txt")
        mezcladas += 1

    yaml = destino / "carga.yaml"
    yaml.write_text(
        "# Generado por scripts/finetune_load.py. No editar a mano.\n"
        f"path: {destino.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: package\n",
        encoding="utf-8",
    )
    print(
        f"dataset: {reparto['train']} nuevas + {mezcladas} de fábrica en entrenamiento, "
        f"{reparto['val']} en validación (vídeo '{val_video}')"
    )
    return yaml


def _liberar() -> None:
    """Soltar la GPU entre etapas.

    Hace falta de verdad y costó una corrida entera: en una 3050 de 6 GB, entrenar
    justo después de dos validaciones revienta con `unable to find an engine to
    execute this computation`, que no dice «sin memoria» pero lo es — cuDNN no
    encuentra sitio para su espacio de trabajo. El modelo de la evaluación ya no se
    referencia, pero la caché del asignador de PyTorch sigue reservada.
    """
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evaluar(pesos, yaml_cliente: Path, device, batch: int) -> dict:
    """mAP en los dos frentes: las cargas del cliente y las de fábrica."""
    from ultralytics import YOLO

    modelo = YOLO(str(pesos))
    cliente = modelo.val(
        data=str(yaml_cliente), device=device, batch=batch, verbose=False, plots=False
    )
    fabrica = modelo.val(
        data=str(BASE_DATASET / "package-seg.yaml"),
        device=device,
        batch=batch,
        verbose=False,
        plots=False,
    )
    salida = {
        "cliente_mascara_map50": round(float(cliente.seg.map50), 4),
        "cliente_caja_map50": round(float(cliente.box.map50), 4),
        "fabrica_mascara_map50": round(float(fabrica.seg.map50), 4),
        "fabrica_caja_map50": round(float(fabrica.box.map50), 4),
    }
    del modelo, cliente, fabrica
    _liberar()
    return salida


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", default=str(RAIZ / "data/carga"))
    parser.add_argument("--pesos", default=str(PESOS_BASE))
    parser.add_argument("--val-video", default=None, help="qué vídeo queda fuera del entrenamiento")
    parser.add_argument("--epocas", type=int, default=40)
    parser.add_argument("--base-por-nuevo", type=int, default=BASE_POR_NUEVO)
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="16 es el que usó el entrenamiento base. En la 3050 de 6 GB no cabe mucho más",
    )
    parser.add_argument("--device", default=0)
    parser.add_argument(
        "--incluir-pantalla",
        action="store_true",
        help="ensayar el flujo con capturas de pantalla. El resultado NO es un número",
    )
    args = parser.parse_args()

    datos = Path(args.datos)
    manifiesto, buenas = _cargar(datos)
    if not buenas:
        print("no queda ninguna propuesta válida")
        return 2

    de_pantalla = {f["archivo"] for f in buenas if f.get("fuente") == "pantalla"}
    if de_pantalla and not args.incluir_pantalla:
        print(f"\nME NIEGO: {len(de_pantalla)} fotogramas vienen de una captura de pantalla.")
        print("Llevan el esqueleto del panel pintado encima de la persona y de la carga, así")
        print("que el modelo aprendería a buscar líneas de colores y la validación, que lleva")
        print("las mismas líneas, no lo delataría. Graba con scripts/record_loads.py.")
        print("Para ensayar el flujo igualmente: --incluir-pantalla")
        return 2
    if de_pantalla:
        print(f"\nAVISO: {len(de_pantalla)} fotogramas son capturas de pantalla.")
        print("       Esto es un ENSAYO del flujo. El número que salga no describe nada.\n")

    videos = sorted({f["video"] for f in buenas})
    val_video = args.val_video or (videos[-1] if len(videos) > 1 else None)
    if val_video is None:
        print(f"\nME NIEGO: todas las propuestas vienen de un solo vídeo ({videos[0]}).")
        print("Partir por fotograma dejaría casi la misma imagen a los dos lados y el")
        print("número saldría inflado. Hacen falta al menos dos vídeos.")
        return 2
    if val_video not in videos:
        print(f"el vídeo '{val_video}' no está entre {videos}")
        return 2

    yaml = _construir(datos, buenas, val_video, args.base_por_nuevo)

    print("\n-- antes de afinar")
    antes = _evaluar(args.pesos, yaml, args.device, args.batch)
    print("   ", antes)

    from ultralytics import YOLO

    modelo = YOLO(args.pesos)
    resultado = modelo.train(
        data=str(yaml),
        epochs=args.epocas,
        batch=args.batch,
        device=args.device,
        project=str(datos / "runs"),
        name="afinado",
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    pesos_nuevos = Path(resultado.save_dir) / "weights/best.pt"
    del modelo
    _liberar()

    print("\n-- después de afinar")
    despues = _evaluar(pesos_nuevos, yaml, args.device, args.batch)
    print("   ", despues)

    artefacto = {
        "fecha": date.today().isoformat(),
        "ensayo_con_capturas_de_pantalla": bool(de_pantalla),
        "fuente_del_material": manifiesto["fuente"],
        "pesos_base": str(args.pesos),
        "pesos_afinados": str(pesos_nuevos),
        "epocas": args.epocas,
        "base_por_nuevo": args.base_por_nuevo,
        "videos": videos,
        "video_de_validacion": val_video,
        "fotogramas_validados": len(buenas),
        "antes": antes,
        "despues": despues,
    }
    salida = RAIZ / "artifacts" / f"carga-afinado-{date.today().isoformat()}.json"
    salida.write_text(json.dumps(artefacto, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nartefacto: {salida}")

    print("\nCÓMO LEERLO:")
    print("  cliente ↑  y  fabrica ≈  → el afinado sirvió y no olvidó")
    print("  cliente ↑  y  fabrica ↓  → olvido catastrófico: menos épocas o más base")
    print("  cliente ≈              → no aprendió; mira si las etiquetas son correctas")
    if len(buenas) < 60:
        print(f"\nY con {len(buenas)} fotogramas validados, ninguna de esas lecturas es una tasa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
