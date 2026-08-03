"""
ai_detector.py
================
Orquestador principal. Combina, en capas de decisión, las siguientes fuentes
de evidencia (en este orden de prioridad):

  1) metadata_analysis  -> ¿Hay una etiqueta/firma de una herramienta de IA
     conocida (Suno, Gemini, Mureka, AIVA, Mubert, Soundraw, ElevenLabs...)?
     Si sí -> fuerte evidencia de IA (peso dominante).

  2) shazam_analysis -> Si los metadatos no fueron concluyentes, se intenta
     reconocer la canción vía huella acústica (ShazamIO). Si Shazam la
     reconoce como una grabación existente -> fuerte evidencia de humano.

  3) acoustic_and_lyrics_analysis -> Si ninguna de las anteriores es
     concluyente, se analizan las características acústicas (librosa /
     pyAudioAnalysis) y, si hay letra disponible, las heurísticas textuales,
     fusionando ambas señales con lógica difusa (scikit-fuzzy).

El resultado final siempre se combina también un poco con el análisis de
contenido (acústico+letra) para dar más contexto, según los pesos definidos
en `config.WEIGHTS`.
"""
from __future__ import annotations
import os

from config import CLASSIFICATION_THRESHOLDS, WEIGHTS
from utils import clip, safe_round, make_bar

from metadata_analysis import analyze_metadata
from shazam_analysis import analyze_shazam
from acoustic_analysis import analyze_acoustic
from lyrics_analysis import analyze_lyrics
from fuzzy_engine import fuzzy_combine


def _classify(score: float) -> str:
    for threshold, label in CLASSIFICATION_THRESHOLDS:
        if score < threshold:
            return label
    return CLASSIFICATION_THRESHOLDS[-1][1]


def analyze_song(file_path: str, lyrics_text: str | None = None,
                  skip_shazam: bool = False) -> dict:
    """
    Ejecuta el pipeline completo de detección y devuelve el diccionario final
    listo para ser serializado a JSON, siguiendo el esquema solicitado.
    """
    audio_file_name = os.path.basename(file_path)

    # ---------------------------------------------------------------
    # Capa 1: Metadatos
    # ---------------------------------------------------------------
    metadata_result = analyze_metadata(file_path)
    metadata_conclusive = bool(metadata_result.get("is_conclusive"))

    # ---------------------------------------------------------------
    # Capa 2: Shazam (huella acústica) - se evita si el usuario pide saltarla
    # ---------------------------------------------------------------
    if skip_shazam:
        shazam_result = {
            "skipped": True, "recognized": False, "is_conclusive": False,
            "reasoning": "Se omitió la consulta a Shazam por configuración de la petición.",
            "track": None, "_score": None,
        }
    else:
        shazam_result = analyze_shazam(file_path)
    shazam_conclusive = bool(shazam_result.get("is_conclusive"))

    # ---------------------------------------------------------------
    # Capa 3: Análisis acústico + letras (siempre se calcula, ya que aporta
    # contexto adicional aunque las capas 1 y 2 ya sean concluyentes)
    # ---------------------------------------------------------------
    acoustic_result = analyze_acoustic(file_path)
    lyrics_result = analyze_lyrics(lyrics_text)

    fuzzy_result = fuzzy_combine(
        acoustic_score=acoustic_result.get("acoustic_ai_likelihood"),
        lyrics_score=lyrics_result.get("lyrics_ai_likelihood") if lyrics_result.get("available") else None,
    )
    content_ai_score = fuzzy_result["content_ai_score"]

    # ---------------------------------------------------------------
    # Fusión final de capas -> ai_score, decision_layer, decision_reasoning
    # ---------------------------------------------------------------
    if metadata_conclusive:
        decision_layer = "metadata"
        final_score = (
            metadata_result["metadata_ai_score"] * WEIGHTS["metadata_conclusive_metadata_weight"] +
            content_ai_score * WEIGHTS["metadata_conclusive_content_weight"]
        )
        tool = metadata_result.get("ai_tool_detected")
        decision_reasoning = (
            (f"Se detectó la firma de la herramienta de IA '{tool}' en los metadatos del archivo. "
             if tool else "Se detectó una pista genérica de generación por IA en los metadatos. ")
            + "Esta evidencia es determinante y domina el puntaje final, complementada levemente "
              f"por el análisis de contenido acústico/lírico (content_ai_score={content_ai_score})."
        )
    elif shazam_conclusive:
        decision_layer = "shazam"
        final_score = (
            shazam_result["_score"] * WEIGHTS["shazam_conclusive_shazam_weight"] +
            content_ai_score * WEIGHTS["shazam_conclusive_content_weight"]
        )
        track = shazam_result.get("track") or {}
        decision_reasoning = (
            f"No se hallaron firmas de IA en los metadatos. Shazam reconoció la canción como "
            f"\"{track.get('title')}\" de {track.get('artist')}, lo cual es fuerte evidencia de "
            f"que es una grabación humana existente. Se combina con el análisis de contenido "
            f"(content_ai_score={content_ai_score}) para el puntaje final."
        )
    else:
        decision_layer = "acoustic_lyrics_fuzzy"
        final_score = content_ai_score
        decision_reasoning = (
            "No se hallaron firmas de herramientas de IA en los metadatos ni se pudo confirmar "
            "la canción vía Shazam (no reconocida o servicio no disponible). El puntaje final se "
            "basa en el análisis de características acústicas (ZCR, centroide/ancho de banda/"
            "contraste/rolloff espectral, flujo espectral, MFCC, RMSE) y, si estaba disponible, "
            "en el análisis heurístico de la letra, fusionados mediante lógica difusa "
            f"({fuzzy_result['method']})."
        )

    final_score = clip(final_score)
    classification = _classify(final_score)

    # ---------------------------------------------------------------
    # Construcción de la respuesta final según el esquema solicitado
    # ---------------------------------------------------------------
    metadata_out = {k: v for k, v in metadata_result.items()}
    shazam_out = {k: v for k, v in shazam_result.items() if not k.startswith("_")}

    acoustic_features = acoustic_result["acoustic_features"]
    summary_blocks = acoustic_result["summary_blocks"]

    response = {
        "audio_file": audio_file_name,
        "scale": {"0": "Humano", "100": "Generado por IA"},
        "ai_score": safe_round(final_score, 2),
        "ai_score_bar": make_bar(final_score),
        "classification": classification,
        "decision_layer": decision_layer,
        "decision_reasoning": decision_reasoning,

        "shazam_analysis": shazam_out,
        "metadata_analysis": metadata_out,

        "acoustic_and_lyrics_analysis": {
            "acoustic_features": acoustic_features,
            "lyrics_analysis": lyrics_result,
            "fuzzy_inference": fuzzy_result,
        },

        # Bloques resumidos de nivel superior (para fácil renderizado en UI / app móvil)
        "Zero Crossing Rate": summary_blocks["Zero Crossing Rate"],
        "Spectral Centroid": summary_blocks["Spectral Centroid"],
        "Spectral Bandwidth": summary_blocks["Spectral Bandwidth"],
        "Spectral Contrast": summary_blocks["Spectral Contrast"],
        "Spectral Rolloff": summary_blocks["Spectral Rolloff"],
        "Spectral Flux": summary_blocks["Spectral Flux"],
        "MFCCs": summary_blocks["MFCCs"],
        "RMSE": summary_blocks["RMSE"],
    }
    return response
