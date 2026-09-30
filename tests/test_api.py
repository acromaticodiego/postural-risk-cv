"""Que el servidor responda y que el JSON tenga lo que la página pide.

La prueba que importa es la del CONTRATO. El panel es JavaScript leyendo un JSON
que produce Python, y ahí no hay compilador que avise: si el pipeline deja de
escribir un campo o le cambia el nombre, la página no falla — dibuja `undefined` en
una tabla, o una tarjeta vacía, y eso se descubre grabando el vídeo.

Así que la lista de campos que el navegador usa está escrita aquí, y se comprueba
contra los datos de verdad.

    .\\.venv\\Scripts\\python.exe -m tests.test_api
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RAIZ = Path(__file__).resolve().parents[1]
WEB = RAIZ / "src/api/web/index.html"

# Lo que el JavaScript lee de cada nivel del JSON. Si se añade un campo a la
# página, se añade aquí; si el backend deja de mandarlo, esto cae.
CAMPOS_RAIZ = ("generado", "hz", "puestos")
CAMPOS_PUESTO = (
    "id", "name", "configured", "config", "shifts", "seconds", "seconds_at_risk",
    "seconds_by_level", "tasks", "recommendation", "events", "total_events",
    "worst_lift",
)
CAMPOS_CONFIG = ("load_kg", "coupling", "risk_threshold")
CAMPOS_TAREA = (
    "task", "seconds_total", "seconds_at_risk", "share_of_risk",
    "median_reba", "dominant_component",
)
CAMPOS_EVENTO = (
    "id", "shift", "start_seconds", "duration_seconds", "peak_reba", "peak_level",
    "dominant_component", "task", "keypoints", "reba", "components", "niosh",
)
COMPONENTES = ("trunk", "neck", "legs", "upper_arm", "lower_arm")
# Lo que la tarjeta NIOSH del panel lee de cada análisis de levantamiento.
CAMPOS_NIOSH = (
    "horizontal_cm", "vertical_cm", "travel_cm", "frequency_per_min", "multipliers",
    "recommended_weight_kg", "lifting_index", "scale_estimated", "frequency_estimated",
)
MULTIPLICADORES = ("horizontal", "vertical", "recorrido", "asimetria", "frecuencia", "agarre")


def _cliente():
    from fastapi.testclient import TestClient

    from src.api.server import app

    return TestClient(app)


def test_server_serves_the_page() -> None:
    respuesta = _cliente().get("/")
    assert respuesta.status_code == 200
    assert "Riesgo postural" in respuesta.text


def test_panel_endpoint_answers() -> None:
    respuesta = _cliente().get("/api/panel")
    assert respuesta.status_code == 200, (
        f"{respuesta.status_code}: genera los datos con scripts/build_demo_data.py"
    )
    datos = respuesta.json()
    for campo in CAMPOS_RAIZ:
        assert campo in datos, f"falta {campo} en la raiz del panel"
    assert datos["puestos"], "el panel no trae ningun puesto"


def test_json_has_every_field_the_page_reads() -> None:
    """El contrato entre Python y el navegador, campo a campo."""
    datos = _cliente().get("/api/panel").json()
    fallos = []
    for puesto in datos["puestos"]:
        for campo in CAMPOS_PUESTO:
            if campo not in puesto:
                fallos.append(f"puesto {puesto.get('id')}: falta {campo}")
        for campo in CAMPOS_CONFIG:
            if campo not in puesto.get("config", {}):
                fallos.append(f"puesto {puesto.get('id')}: falta config.{campo}")
        for tarea in puesto.get("tasks", []):
            for campo in CAMPOS_TAREA:
                if campo not in tarea:
                    fallos.append(f"tarea de {puesto.get('id')}: falta {campo}")
            break
        for evento in puesto.get("events", []):
            for campo in CAMPOS_EVENTO:
                if campo not in evento:
                    fallos.append(f"evento de {puesto.get('id')}: falta {campo}")
            for componente in COMPONENTES:
                if componente not in evento["components"][0]:
                    fallos.append(f"componentes del evento: falta {componente}")
            break
    assert not fallos, "el JSON no cumple lo que la pagina lee: " + "; ".join(fallos[:6])


def test_niosh_analysis_is_complete_where_it_exists() -> None:
    """La tarjeta NIOSH lee muchos campos y ninguno da error si falta: dibuja
    `undefined` en una recomendación que habla de kilos."""
    datos = _cliente().get("/api/panel").json()
    fallos, analizados = [], 0
    for puesto in datos["puestos"]:
        candidatos = [e["niosh"] for e in puesto["events"] if e.get("niosh")]
        if puesto.get("worst_lift"):
            candidatos.append(puesto["worst_lift"])
        for analisis in candidatos:
            analizados += 1
            for campo in CAMPOS_NIOSH:
                if campo not in analisis:
                    fallos.append(f"{puesto['id']}: falta niosh.{campo}")
            for m in MULTIPLICADORES:
                if m not in analisis.get("multipliers", {}):
                    fallos.append(f"{puesto['id']}: falta el multiplicador {m}")
    assert analizados, "ningun puesto trajo analisis NIOSH: la tarjeta no se veria nunca"
    assert not fallos, "; ".join(fallos[:5])


def test_niosh_only_applies_to_lifts() -> None:
    """La ecuación mide levantar una carga. Aplicarla a estar de pie en mala postura
    daría un índice sin significado, y el panel lo mostraría igual de creíble."""
    datos = _cliente().get("/api/panel").json()
    for puesto in datos["puestos"]:
        for evento in puesto["events"]:
            if evento.get("niosh"):
                assert "pick-up" in evento["task"] or "place" in evento["task"], (
                    f"evento {evento['id']} tiene NIOSH y su tarea es {evento['task']!r}"
                )


def test_recommended_weight_is_never_a_free_pass() -> None:
    """El peso recomendado no puede superar la constante de la norma (23 kg).

    Es la cota que impide que un error de escala —una estatura mal declarada, por
    ejemplo— produzca un 'puedes levantar 60 kg' con toda naturalidad.
    """
    datos = _cliente().get("/api/panel").json()
    for puesto in datos["puestos"]:
        peor = puesto.get("worst_lift")
        if peor:
            assert 0 <= peor["recommended_weight_kg"] <= 23.0, peor["recommended_weight_kg"]


def test_replay_arrays_line_up() -> None:
    """Los tres arrays de un replay tienen que tener la misma longitud.

    El navegador los indexa con el mismo contador, así que uno más corto no da
    error: da `undefined` a mitad de la reproducción.
    """
    datos = _cliente().get("/api/panel").json()
    for puesto in datos["puestos"]:
        for evento in puesto["events"]:
            n = len(evento["keypoints"])
            assert len(evento["reba"]) == n and len(evento["components"]) == n, (
                f"evento {evento['id']}: {n} esqueletos, {len(evento['reba'])} puntajes, "
                f"{len(evento['components'])} componentes"
            )
            assert len(evento["keypoints"][0]) == 17, "no son 17 articulaciones"
            assert len(evento["keypoints"][0][0]) == 2, "cada articulacion son 2 numeros"


def test_skeletons_fit_in_the_canvas_range() -> None:
    """Llegan normalizados a 0..1: si se salen, la pagina dibuja fuera del lienzo."""
    datos = _cliente().get("/api/panel").json()
    for puesto in datos["puestos"]:
        for evento in puesto["events"][:3]:
            for fotograma in evento["keypoints"]:
                for x, y in fotograma:
                    assert -0.1 <= x <= 1.1 and -0.1 <= y <= 1.1, (
                        f"evento {evento['id']}: punto fuera de rango ({x}, {y})"
                    )


def test_no_image_data_reaches_the_browser() -> None:
    """La garantia de privacidad, comprobada en el borde del sistema.

    Aunque el pipeline no guarde imagenes, lo que hace verdad la promesa es que no
    salga ninguna por la API. Si alguien anadiera un campo con una imagen en base64
    para 'mejorar la demo', esto cae.
    """
    crudo = _cliente().get("/api/panel").text
    for sospechoso in ("data:image", "base64", ".jpg", ".png", "frame_", "image"):
        assert sospechoso not in crudo, f"el panel lleva algo con pinta de imagen: {sospechoso}"


def test_page_does_not_depend_on_a_cdn() -> None:
    """Sin recursos externos: una libreria remota puede fallar justo en la toma."""
    html = WEB.read_text(encoding="utf-8")
    externos = re.findall(r'(?:src|href)\s*=\s*["\'](https?://[^"\']+)', html)
    assert not externos, f"la pagina carga recursos externos: {externos}"


def test_missing_workstation_is_a_404() -> None:
    assert _cliente().get("/api/workstations/NO-EXISTE").status_code == 404
    datos = _cliente().get("/api/panel").json()
    primero = datos["puestos"][0]["id"]
    assert _cliente().get(f"/api/workstations/{primero}").status_code == 200


if __name__ == "__main__":
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fallos = 0
    for prueba in pruebas:
        try:
            prueba()
            print(f"  ok   {prueba.__name__}")
        except AssertionError as e:
            fallos += 1
            print(f"  FALLA {prueba.__name__}: {e}")
    print(f"\n{len(pruebas) - fallos}/{len(pruebas)}")
    sys.exit(1 if fallos else 0)
