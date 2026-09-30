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

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
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
