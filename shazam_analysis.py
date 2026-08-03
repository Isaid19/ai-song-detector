"""
shazam_analysis.py
====================
Usa la librería `shazamio` para intentar reconocer la canción a partir del
propio audio (huella acústica), tal como lo haría la app de Shazam.

Si la canción es reconocida y se obtienen datos coherentes de artista/título/
álbum/año, esto es evidencia fuerte de que se trata de una grabación humana
ya publicada comercialmente (reduce el ai_score final).

Si no es reconocida (o el servicio no está disponible/hay error de red),
se marca `recognized=False` (o `skipped=True`) y el flujo de decisión pasa a
depender del análisis de metadatos y/o del análisis acústico + letras.
"""
from __future__ import annotations
import asyncio

try:
    from shazamio import Shazam
    SHAZAMIO_AVAILABLE = True
except ImportError:  # pragma: no cover
    SHAZAMIO_AVAILABLE = False

from config import SHAZAM_RECOGNIZED_BASE_SCORE

# Tiempo máximo (segundos) que se espera la respuesta de Shazam antes de
# considerar el servicio no disponible (evita bloquear la API por mucho tiempo).
SHAZAM_TIMEOUT_SECONDS = 20


async def _recognize_async(file_path: str) -> dict:
    shazam = Shazam()
    return await shazam.recognize(file_path)


def recognize_track(file_path: str) -> dict:
    """
    Ejecuta el reconocimiento de forma síncrona (para poder usarse
    fácilmente desde Flask), envolviendo la llamada asíncrona de shazamio.
    """
    if not SHAZAMIO_AVAILABLE:
        return {"_skipped": True, "_reason": "La librería 'shazamio' no está instalada."}

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("loop cerrado")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            asyncio.wait_for(_recognize_async(file_path), timeout=SHAZAM_TIMEOUT_SECONDS)
        )
        return {"_skipped": False, "_reason": None, "_raw": result}
    except asyncio.TimeoutError:
        return {"_skipped": True, "_reason": "Tiempo de espera agotado al consultar el servicio de Shazam."}
    except Exception as exc:  # errores de red, formato de audio no soportado, etc.
        return {"_skipped": True, "_reason": f"No fue posible consultar Shazam: {exc}"}


def analyze_shazam(file_path: str) -> dict:
    """
    Punto de entrada del módulo. Devuelve un diccionario con la forma
    esperada por el esquema JSON de salida ("shazam_analysis"), además de
    un campo interno `_score` usado por el orquestador para combinar capas.
    """
    outcome = recognize_track(file_path)

    if outcome.get("_skipped"):
        return {
            "skipped": True,
            "recognized": False,
            "is_conclusive": False,
            "reasoning": outcome.get("_reason") or "No se pudo ejecutar el reconocimiento con Shazam.",
            "track": None,
            "_score": None,
        }

    raw = outcome.get("_raw") or {}
    track = raw.get("track")

    if not track:
        return {
            "skipped": False,
            "recognized": False,
            "is_conclusive": False,
            "reasoning": (
                "El audio fue enviado a Shazam pero no se encontró coincidencia con "
                "ninguna grabación conocida. Esto no confirma ni descarta el uso de IA; "
                "se recurrirá al análisis de metadatos y/o de contenido."
            ),
            "track": None,
            "_score": None,
        }

    title = track.get("title")
    subtitle = track.get("subtitle")  # normalmente el artista
    sections = track.get("sections", []) or []
    album = None
    year = None
    genre = track.get("genres", {}).get("primary") if isinstance(track.get("genres"), dict) else None

    for section in sections:
        if section.get("type") == "SONG":
            for item in section.get("metadata", []) or []:
                title_meta = (item.get("title") or "").lower()
                if "album" in title_meta:
                    album = item.get("text")
                if "released" in title_meta or "year" in title_meta:
                    year = item.get("text")

    recognized_data = {
        "title": title,
        "artist": subtitle,
        "album": album,
        "year": year,
        "genre": genre,
        "shazam_track_id": track.get("key"),
        "shazam_url": track.get("url"),
    }

    return {
        "skipped": False,
        "recognized": True,
        "is_conclusive": True,
        "reasoning": (
            f"Shazam reconoció la pista como \"{title}\" de {subtitle}"
            + (f" (álbum: {album})" if album else "")
            + (f" (año: {year})" if year else "")
            + ". Al tratarse de una grabación ya identificada/publicada, se reduce "
              "considerablemente la probabilidad de que sea una canción generada por IA."
        ),
        "track": recognized_data,
        "_score": SHAZAM_RECOGNIZED_BASE_SCORE,
    }
