"""El servidor del panel. Sirve la página y los datos ya calculados.

La separación es deliberada y es la que tendría un despliegue real: el pipeline
procesa las grabaciones cuando toca —de noche, por lotes, con GPU— y el servidor
solo lee el resultado. Calcular en la petición ataría la latencia del panel a la
velocidad de la GPU y haría que abrir una pantalla costara lo que cuesta procesar
un turno.

Lo que viaja al navegador son esqueletos y números. No hay ningún endpoint que
pueda devolver una imagen de una persona, porque no existe ninguna guardada.

    .\\.venv\\Scripts\\python.exe -m uvicorn src.api.server:app --port 8000
    y abrir http://127.0.0.1:8000/
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

RAIZ = Path(__file__).resolve().parents[2]
PANEL = RAIZ / "artifacts/demo/panel.json"
WEB = Path(__file__).resolve().parent / "web"

app = FastAPI(title="Riesgo postural por vision", version="0.1.0")

_cache: dict | None = None


def _panel() -> dict:
    """Lee el panel una vez y lo deja en memoria.

    Son megabytes de esqueletos y no cambian entre peticiones: releerlos en cada
    una convertiría el disco en el cuello de botella de una pantalla que solo
    muestra tablas.
    """
    global _cache
    if _cache is None:
        if not PANEL.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "No hay datos generados todavia. Corre "
                    "python scripts/build_demo_data.py"
                ),
            )
        _cache = json.loads(PANEL.read_text(encoding="utf-8"))
    return _cache


@app.get("/api/panel")
def panel() -> JSONResponse:
    """Todo el panel de una vez: puestos, informes y los replays de los peores eventos."""
    return JSONResponse(_panel())


@app.get("/api/workstations/{workstation_id}")
def workstation(workstation_id: str) -> JSONResponse:
    for puesto in _panel()["puestos"]:
        if puesto["id"] == workstation_id:
            return JSONResponse(puesto)
    raise HTTPException(status_code=404, detail=f"no existe el puesto {workstation_id}")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


# --- el modo en vivo ----------------------------------------------------------


@app.websocket("/ws/live")
async def live(socket: WebSocket) -> None:
    """La cámara, procesada a 10 Hz, emitida fotograma a fotograma.

    La imagen de la cámara viaja por aquí y NO SE GUARDA en ningún sitio: es la
    vista de instalación, la que necesita quien monta la cámara para comprobar el
    encuadre. Lo que el sistema conserva de un turno son los esqueletos, y eso se
    ve en la columna de al lado de la misma pantalla.

    Se procesa a 10 Hz aunque la cámara dé más, porque el modelo se entrenó con
    ventanas de 2 s a esa tasa. Acelerar la entrada le daría el gesto comprimido
    sin dar ningún error.
    """
    await socket.accept()
    sesion = None
    try:
        from ..live.session import TARGET_HZ, LiveSession
        from ..product.workstation import WorkstationConfig

        parametros = socket.query_params
        config = WorkstationConfig(
            id=parametros.get("id", "EN-VIVO"),
            name=parametros.get("name", "Camara en vivo"),
            load_kg=float(parametros["load_kg"]) if parametros.get("load_kg") else None,
            coupling=parametros.get("coupling") or None,
        )
        sesion = LiveSession(config, camera=int(parametros.get("camera", 0)))
        intervalo = 1.0 / TARGET_HZ

        while True:
            comenzado = time.perf_counter()
            estado = await run_in_threadpool(sesion.step)
            if estado is None:
                await socket.send_json({"error": "la camara dejo de dar imagen"})
                break
            await socket.send_json(estado)
            # Se duerme lo que falte para el siguiente ciclo, no un tiempo fijo: si
            # un fotograma tarda más de lo previsto, esperar igual iría acumulando
            # retraso y el vídeo se vería a cámara lenta.
            resto = intervalo - (time.perf_counter() - comenzado)
            if resto > 0:
                await asyncio.sleep(resto)
    except WebSocketDisconnect:
        pass
    except Exception as error:  # noqa: BLE001
        try:
            await socket.send_json({"error": str(error)})
        except Exception:  # noqa: BLE001
            pass
    finally:
        if sesion is not None:
            sesion.close()
