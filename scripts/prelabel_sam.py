"""SAM pre-etiqueta la carga, guiado por el esqueleto, y una persona valida.

EL FLUJO, Y POR QUÉ TIENE UN HUMANO EN MEDIO. Etiquetar máscaras a mano es lo que
mata este tipo de proyecto: son horas por cada cien imágenes. Aquí SAM las propone
—el esqueleto ya dice dónde está la carga— y el trabajo de la persona pasa de
DIBUJAR a MIRAR Y DESCARTAR, que es un segundo por imagen en vez de un minuto.

El humano no es un trámite. Un pre-etiquetado es una propuesta, y las propuestas
malas de SAM son plausibles: segmenta algo, con un contorno limpio, y a veces ese
algo es el sofá del fondo. Los filtros de `prelabel.py` quitan las que se pueden
descartar por geometría; las que quedan hay que verlas. Entrenar con lo que salga de
aquí sin que nadie lo haya mirado es la forma más directa de obtener un número
limpio sobre material falso.

AQUÍ SE ELIGE EL CONJUNTO, y no en la extracción. SAM se pasa por TODOS los
fotogramas candidatos, y la selección de los más variados en postura se hace entre
los que SOBREVIVEN a los filtros. Al revés —elegir variedad primero y segmentar
después— el conjunto se llena de fotogramas sin ninguna carga en las manos: medido
sobre un vídeo real, 11 de 12 elegidos por variedad no tenían nada que segmentar.
Una postura rara es exactamente lo que hace alguien cuando NO está cargando nada.

La hoja de contactos sale como una página HTML sin un solo recurso externo, por la
misma razón que el panel: se abre con doble clic, funciona sin servidor y sin
internet, y lo que se marque se guarda en el navegador para poder dejarlo a medias.

    .\\.venv\\Scripts\\python.exe scripts\\prelabel_sam.py --datos data\\carga
    # y abrir data\\carga\\revision.html
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.load.prelabel import (  # noqa: E402
    build_prompt,
    judge_mask,
    mask_to_polygon,
    pick_varied,
    polygon_to_yolo_label,
    pose_vector,
)

RAIZ = Path(__file__).resolve().parents[1]
PESOS_SAM = RAIZ / "mobile_sam.pt"

VERDE = (120, 220, 120)
ROJO = (90, 90, 230)


def _dibujar(imagen, poligono, aviso) -> np.ndarray:
    """La miniatura de revisión: el contorno de la máscara y los puntos de aviso.

    Se pinta el CONTORNO y no la máscara rellena a propósito: una mancha de color
    tapa justo el borde, que es lo único que hay que juzgar para decir si la máscara
    cae sobre la carga o se come un trozo del fondo.

    Y se dibuja el POLÍGONO ya simplificado, no la máscara en bruto, porque es lo que
    de verdad se va a entrenar. Validar el contorno fino y entrenar con el
    simplificado dejaría sin mirar justamente el paso que pierde información.
    """
    salida = imagen.copy()
    if poligono is not None:
        cv2.polylines(salida, [poligono.astype(np.int32)], True, VERDE, 2)
    for (x, y), etiqueta in zip(aviso.points, aviso.labels):
        color = (255, 220, 60) if etiqueta else ROJO
        cv2.circle(salida, (int(x), int(y)), 4, color, -1)
        cv2.circle(salida, (int(x), int(y)), 5, (20, 20, 20), 1)
    return salida


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", default=str(RAIZ / "data/carga"))
    parser.add_argument("--pesos-sam", default=str(PESOS_SAM))
    parser.add_argument("--por-video", type=int, default=40, help="cuántos se validan a mano")
    parser.add_argument("--min-separacion-s", type=float, default=1.0)
    parser.add_argument("--device", default=0)
    args = parser.parse_args()

    datos = Path(args.datos)
    manifiesto = json.loads((datos / "manifiesto.json").read_text(encoding="utf-8"))
    carpeta_frames = datos / "frames"
    carpeta_labels = datos / "labels"
    carpeta_revision = datos / "revision"
    carpeta_labels.mkdir(parents=True, exist_ok=True)
    carpeta_revision.mkdir(parents=True, exist_ok=True)

    from ultralytics import SAM

    sam = SAM(args.pesos_sam)

    aceptadas, rechazos = [], []
    total = len(manifiesto["frames"])
    for n, entrada in enumerate(manifiesto["frames"], 1):
        if n % 50 == 0 or n == total:
            print(f"   SAM {n}/{total}")
        imagen = cv2.imread(str(carpeta_frames / entrada["archivo"]))
        if imagen is None:
            continue
        kp = np.array(entrada["keypoints"], dtype=np.float32)
        scores = np.array(entrada["scores"], dtype=np.float32)

        aviso = build_prompt(kp, scores)
        if aviso is None:
            rechazos.append({**_resumen(entrada), "motivo": "sin puntos de aviso utilizables"})
            continue

        resultado = sam.predict(
            imagen, points=aviso.points, labels=aviso.labels, device=args.device, verbose=False
        )[0]
        if resultado.masks is None or not len(resultado.masks.data):
            rechazos.append({**_resumen(entrada), "motivo": "SAM no devolvió máscara"})
            continue

        mascara = resultado.masks.data[0].cpu().numpy().astype(np.uint8)
        veredicto = judge_mask(mascara, kp, scores)
        poligono = mask_to_polygon(mascara)
        if veredicto.ok and poligono is None:
            veredicto = type(veredicto)(False, "la máscara no da un polígono")

        registro = {**_resumen(entrada), "motivo": veredicto.reason}
        if not veredicto.ok:
            rechazos.append({**registro, "_poligono": poligono, "_aviso": aviso})
            continue
        aceptadas.append(
            {
                **registro,
                "vertices": len(poligono),
                "_poligono": poligono,
                "_aviso": aviso,
                "_entrada": entrada,
            }
        )

    # La variedad se busca AQUÍ, entre los fotogramas donde SAM encontró carga.
    propuestas = []
    for video in {a["video"] for a in aceptadas}:
        delvideo = [a for a in aceptadas if a["video"] == video]
        vectores = [
            pose_vector(
                np.array(a["_entrada"]["keypoints"], dtype=np.float32),
                np.array(a["_entrada"]["box"]),
            )
            for a in delvideo
        ]
        tiempos = [a["t_s"] for a in delvideo]
        for i in pick_varied(vectores, tiempos, args.por_video, args.min_separacion_s):
            propuestas.append(delvideo[i])
    propuestas.sort(key=lambda p: (p["video"], p["t_s"]))

    for p in propuestas:
        entrada = p["_entrada"]
        etiqueta = polygon_to_yolo_label(p["_poligono"], entrada["ancho"], entrada["alto"])
        (carpeta_labels / f"{Path(entrada['archivo']).stem}.txt").write_text(
            etiqueta + "\n", encoding="utf-8"
        )
    _miniaturas(propuestas, carpeta_frames, carpeta_revision)
    # De las rechazadas basta una muestra: están para poder ver si los filtros se
    # pasan de severos, no para que nadie las revise una a una.
    muestra_rechazos = rechazos[:: max(1, len(rechazos) // 24)][:24]
    _miniaturas(muestra_rechazos, carpeta_frames, carpeta_revision)

    sin_elegir = len(aceptadas) - len(propuestas)
    salida = {
        "creado": date.today().isoformat(),
        "fuente": manifiesto["fuente"],
        "aceptadas_por_los_filtros": len(aceptadas),
        "elegidas_para_validar": len(propuestas),
        "aceptadas_sin_elegir": sin_elegir,
        "propuestas": [_sin_privados(p) for p in propuestas],
        "rechazos": [_sin_privados(r) for r in rechazos],
        "rechazos_con_miniatura": [r["archivo"] for r in muestra_rechazos],
    }
    (datos / "prelabel.json").write_text(
        json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _escribir_revision(datos, salida)

    print(f"\n{len(aceptadas)} de {total} fotogramas pasan los filtros")
    print(f"{len(propuestas)} elegidas para validar a mano ({sin_elegir} aceptadas sin elegir)")
    if rechazos:
        print("\nrechazadas por los filtros:")
        cuenta: dict[str, int] = {}
        for r in rechazos:
            clave = r["motivo"].split("(")[0].strip()
            cuenta[clave] = cuenta.get(clave, 0) + 1
        for clave, n in sorted(cuenta.items(), key=lambda p: -p[1]):
            print(f"  {n:4d}  {clave}")
    print(f"\nabre:  {datos / 'revision.html'}")
    return 0


def _miniaturas(registros, carpeta_frames: Path, carpeta_revision: Path) -> None:
    """Escribe la miniatura de revisión de cada registro que tenga contorno."""
    for r in registros:
        imagen = cv2.imread(str(carpeta_frames / r["archivo"]))
        if imagen is None:
            continue
        cv2.imwrite(
            str(carpeta_revision / r["archivo"]),
            _dibujar(imagen, r.get("_poligono"), r["_aviso"]),
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )


def _sin_privados(registro: dict) -> dict:
    return {k: v for k, v in registro.items() if not k.startswith("_")}


def _resumen(entrada: dict) -> dict:
    return {
        "archivo": entrada["archivo"],
        "video": entrada["video"],
        "fuente": entrada["fuente"],
        "t_s": entrada["t_s"],
    }


def _escribir_revision(datos: Path, salida: dict) -> None:
    con_miniatura = set(salida["rechazos_con_miniatura"])
    propuestas = json.dumps(salida["propuestas"], ensure_ascii=False)
    rechazos = json.dumps(
        [r for r in salida["rechazos"] if r["archivo"] in con_miniatura], ensure_ascii=False
    )
    aviso = ""
    if salida["fuente"] == "pantalla":
        aviso = (
            '<p class="alerta">Este material está declarado como <b>captura de '
            "pantalla</b>: lleva el esqueleto pintado encima. Sirve para ensayar el "
            "flujo y <b>no</b> para afinar el modelo.</p>"
        )
    html = PLANTILLA.replace("{{AVISO}}", aviso)
    html = html.replace("{{PROPUESTAS}}", propuestas).replace("{{RECHAZOS}}", rechazos)
    (datos / "revision.html").write_text(html, encoding="utf-8")


PLANTILLA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Validar el pre-etiquetado de la carga</title>
<style>
/* Sin CDN, igual que el panel: esto se abre con doble clic y sin internet. */
:root{--fondo:#0f1214;--panel:#171b1f;--borde:#2a3036;--texto:#e8eaec;--suave:#8d979f;
      --ok:#7dbf6e;--mal:#e34b4b;--acento:#5aa9e6}
*{box-sizing:border-box}
body{margin:0;background:var(--fondo);color:var(--texto);
  font:14px/1.5 "Segoe UI",system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);
  border-bottom:1px solid var(--borde);padding:16px 24px;display:flex;align-items:center;gap:18px}
h1{margin:0;font-size:16px;font-weight:600}
.cuenta{color:var(--suave);font-size:13px}
.cuenta b{color:var(--texto)}
button{font:inherit;color:var(--texto);background:#222a30;border:1px solid var(--borde);
  border-radius:6px;padding:7px 14px;cursor:pointer}
button:hover{border-color:var(--acento)}
main{max-width:1240px;margin:0 auto;padding:24px}
.alerta{background:#3a2626;border:1px solid #6b3b3b;border-radius:8px;padding:12px 16px}
h2{font-size:14px;font-weight:600;margin:28px 0 6px}
h2 small{color:var(--suave);font-weight:400}
.rejilla{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}
.tarjeta{background:var(--panel);border:2px solid var(--borde);border-radius:10px;
  overflow:hidden;cursor:pointer;transition:border-color .12s}
.tarjeta img{width:100%;display:block;background:#000}
.tarjeta .pie{padding:8px 10px;font-size:11px;color:var(--suave);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tarjeta.buena{border-color:var(--ok)}
.tarjeta.mala{border-color:var(--mal);opacity:.45}
.tarjeta.mala .pie::before{content:"DESCARTADA · ";color:var(--mal);font-weight:600}
.fija{cursor:default;border-color:var(--borde);opacity:.7}
.ayuda{color:var(--suave);margin:0 0 18px}
</style></head><body>
<header>
  <h1>Validar el pre-etiquetado de la carga</h1>
  <span class="cuenta"><b id="buenas">0</b> aceptadas · <b id="malas">0</b> descartadas</span>
  <button id="descargar" style="margin-left:auto">Descargar rechazadas.txt</button>
  <button id="limpiar">Empezar de cero</button>
</header>
<main>
{{AVISO}}
<p class="ayuda">Haz clic en una imagen para descartarla. Lo que marques se guarda en
este navegador, así que puedes dejarlo a medias. Al terminar, descarga
<code>rechazadas.txt</code> y guárdalo junto al manifiesto.</p>
<h2>Propuestas <small>— el contorno verde es la máscara; los puntos amarillos y rojos son lo que se le dijo a SAM</small></h2>
<div class="rejilla" id="propuestas"></div>
<h2>Descartadas por los filtros <small>— no hay que hacer nada con ellas, están aquí para poder ver si los filtros se pasan de severos</small></h2>
<div class="rejilla" id="rechazos"></div>
</main>
<script>
const PROPUESTAS = {{PROPUESTAS}};
const RECHAZOS = {{RECHAZOS}};
const CLAVE = "carga-rechazadas";
let rechazadas = new Set(JSON.parse(localStorage.getItem(CLAVE) || "[]"));

function guardar(){
  localStorage.setItem(CLAVE, JSON.stringify([...rechazadas]));
  document.getElementById("buenas").textContent = PROPUESTAS.length - rechazadas.size;
  document.getElementById("malas").textContent = rechazadas.size;
}

function tarjeta(d, interactiva){
  const div = document.createElement("div");
  div.className = "tarjeta " + (interactiva ? "buena" : "fija");
  const img = document.createElement("img");
  img.src = "revision/" + d.archivo;
  img.loading = "lazy";
  const pie = document.createElement("div");
  pie.className = "pie";
  pie.textContent = d.video + " · " + d.t_s + " s · " + d.motivo;
  div.append(img, pie);
  if (interactiva){
    if (rechazadas.has(d.archivo)) div.classList.replace("buena","mala");
    div.onclick = () => {
      if (rechazadas.has(d.archivo)){ rechazadas.delete(d.archivo); div.classList.replace("mala","buena"); }
      else { rechazadas.add(d.archivo); div.classList.replace("buena","mala"); }
      guardar();
    };
  }
  return div;
}

const cp = document.getElementById("propuestas");
PROPUESTAS.forEach(d => cp.append(tarjeta(d, true)));
const cr = document.getElementById("rechazos");
RECHAZOS.forEach(d => cr.append(tarjeta(d, false)));
guardar();

document.getElementById("descargar").onclick = () => {
  const texto = [...rechazadas].sort().join("\\n") + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([texto], {type:"text/plain"}));
  a.download = "rechazadas.txt";
  a.click();
};
document.getElementById("limpiar").onclick = () => {
  if (!confirm("¿Borrar todas las marcas?")) return;
  rechazadas = new Set(); guardar();
  document.querySelectorAll(".tarjeta.mala").forEach(t => t.classList.replace("mala","buena"));
};
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
