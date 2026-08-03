"""
metadata_analysis.py
=====================
Analiza los metadatos (tags) de un archivo de audio para:
  1. Extraer título, artista, álbum, año, género, comentario y encoder.
  2. Buscar firmas/etiquetas de herramientas conocidas de generación musical
     por IA (Suno, Gemini, Mureka, AIVA, Mubert, Soundful, Soundraw, LANDR,
     Beatoven, Moises, Undio, MusicGPT, ElevenLabs, etc.).
  3. Si no hay una herramienta específica, buscar pistas genéricas de IA.

Usa `mutagen`, que soporta MP3 (ID3), FLAC, OGG, M4A/MP4, WAV, entre otros.
"""
from __future__ import annotations
from typing import Optional

from config import AI_TOOL_SIGNATURES, GENERIC_AI_HINTS, METADATA_FIELDS_TO_SCAN, \
    METADATA_TOOL_DETECTED_SCORE, METADATA_GENERIC_HINT_SCORE

try:
    from mutagen import File as MutagenFile
except ImportError:  # pragma: no cover
    MutagenFile = None


def _first(value):
    """mutagen EasyID3/EasyMP4 devuelve listas; toma el primer elemento como texto."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return str(value[0]) if len(value) else None
    return str(value)


def _extract_year(raw_date: Optional[str]) -> Optional[str]:
    if not raw_date:
        return None
    # Fechas tipo "2023-05-10", "2023", "2023-05-10T00:00:00Z"
    digits = "".join(c for c in raw_date[:4] if c.isdigit())
    return digits if len(digits) == 4 else raw_date


def extract_raw_metadata(file_path: str) -> dict:
    """
    Extrae metadatos "amigables" (title/artist/album/...) y también intenta
    capturar campos técnicos crudos (encoder, comentarios, tags libres tipo
    TXXX/©cmt/©too) que muchas herramientas de IA usan para "firmar" el audio.
    """
    result = {
        "title": None, "artist": None, "album": None, "year": None,
        "genre": None, "comment": None, "encoder": None,
        "raw_fields": {},  # todos los pares clave/valor encontrados (para búsqueda de firmas)
    }
    if MutagenFile is None:
        return result

    try:
        easy = MutagenFile(file_path, easy=True)
        raw = MutagenFile(file_path, easy=False)
    except Exception:
        return result

    if easy is not None and easy.tags is not None:
        tags = easy.tags
        result["title"] = _first(tags.get("title"))
        result["artist"] = _first(tags.get("artist"))
        result["album"] = _first(tags.get("album"))
        result["genre"] = _first(tags.get("genre"))
        result["year"] = _extract_year(_first(tags.get("date")) or _first(tags.get("year")))
        result["encoder"] = _first(tags.get("encodedby"))
        for key in tags.keys():
            try:
                result["raw_fields"][key] = _first(tags.get(key))
            except Exception:
                pass

    # Campos crudos específicos por formato (comentarios, encoder, software...)
    if raw is not None and raw.tags is not None:
        try:
            for key, value in raw.tags.items():
                key_l = str(key).lower()
                try:
                    text_value = str(value)
                except Exception:
                    continue
                result["raw_fields"][key] = text_value

                # ID3 comment frames -> "COMM::..."; encoder -> "TSSE"/"TENC"
                if key_l.startswith("comm") and result["comment"] is None:
                    result["comment"] = text_value
                if key_l.startswith("tsse") or key_l.startswith("tenc"):
                    result["encoder"] = result["encoder"] or text_value
                # MP4/M4A: '\xa9too' (encoder), '\xa9cmt' (comment)
                if key_l.endswith("too") and result["encoder"] is None:
                    result["encoder"] = text_value
                if key_l.endswith("cmt") and result["comment"] is None:
                    result["comment"] = text_value
                # Vorbis comments (FLAC/OGG)
                if key_l in ("comment", "description") and result["comment"] is None:
                    result["comment"] = text_value
                if key_l in ("encoder", "encoder-string", "encoding") and result["encoder"] is None:
                    result["encoder"] = text_value
        except Exception:
            pass

    return result


def _search_signatures(field_values: dict) -> dict:
    """
    Recorre todos los valores de metadatos disponibles buscando coincidencias
    con el catálogo de herramientas de IA o con pistas genéricas.
    """
    for field_name, value in field_values.items():
        if not value:
            continue
        value_l = str(value).lower()
        for tool_name, patterns in AI_TOOL_SIGNATURES.items():
            for pattern in patterns:
                if pattern in value_l:
                    return {
                        "ai_tool_detected": tool_name,
                        "generic_ai_hint": False,
                        "matched_field": field_name,
                        "matched_value": value,
                    }
    # Sin herramienta específica: buscar pistas genéricas
    for field_name, value in field_values.items():
        if not value:
            continue
        value_l = str(value).lower()
        for hint in GENERIC_AI_HINTS:
            if hint in value_l:
                return {
                    "ai_tool_detected": None,
                    "generic_ai_hint": True,
                    "matched_field": field_name,
                    "matched_value": value,
                }
    return {
        "ai_tool_detected": None,
        "generic_ai_hint": False,
        "matched_field": None,
        "matched_value": None,
    }


def analyze_metadata(file_path: str) -> dict:
    """
    Punto de entrada del módulo. Devuelve un diccionario con la forma
    esperada por el esquema JSON de salida ("metadata_analysis").
    """
    meta = extract_raw_metadata(file_path)

    # Construir el diccionario de campos "planos" a inspeccionar + los raw_fields
    scan_fields = {
        "title": meta.get("title"),
        "artist": meta.get("artist"),
        "album": meta.get("album"),
        "genre": meta.get("genre"),
        "comment": meta.get("comment"),
        "encoder": meta.get("encoder"),
    }
    scan_fields.update(meta.get("raw_fields", {}))

    match = _search_signatures(scan_fields)

    is_conclusive = match["ai_tool_detected"] is not None or match["generic_ai_hint"] is True

    if match["ai_tool_detected"] is not None:
        score = METADATA_TOOL_DETECTED_SCORE
        reasoning = (
            f"Se encontró la firma de la herramienta de IA '{match['ai_tool_detected']}' "
            f"en el campo de metadatos '{match['matched_field']}' "
            f"(valor: \"{match['matched_value']}\"). Esto es un indicio muy fuerte de "
            f"que la canción fue generada por inteligencia artificial."
        )
    elif match["generic_ai_hint"]:
        score = METADATA_GENERIC_HINT_SCORE
        reasoning = (
            f"No se identificó una herramienta de IA específica del catálogo conocido, "
            f"pero el campo '{match['matched_field']}' contiene una pista genérica de "
            f"generación por IA (valor: \"{match['matched_value']}\")."
        )
    else:
        score = 0.0
        reasoning = (
            "No se encontraron firmas ni etiquetas de herramientas de IA conocidas "
            "(Suno, Gemini, Mureka, AIVA, Mubert, Soundful, Soundraw, LANDR, Beatoven, "
            "Moises, Undio, MusicGPT, ElevenLabs, etc.) ni pistas genéricas de IA en "
            "los metadatos del archivo."
        )

    return {
        "title": meta.get("title"),
        "artist": meta.get("artist"),
        "album": meta.get("album"),
        "year": meta.get("year"),
        "genre": meta.get("genre"),
        "comment": meta.get("comment"),
        "encoder": meta.get("encoder"),
        "ai_tool_detected": match["ai_tool_detected"],
        "generic_ai_hint": match["generic_ai_hint"],
        "matched_field": match["matched_field"],
        "matched_value": match["matched_value"],
        "is_conclusive": is_conclusive,
        "metadata_ai_score": score,
        "reasoning": reasoning,
    }
