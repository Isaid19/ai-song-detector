"""
fuzzy_engine.py
=================
Combina el "acoustic_ai_likelihood" y el "lyrics_ai_likelihood" (cada uno en
escala 0-100) en un único "content_ai_score" usando un sistema de inferencia
difusa (Mamdani) implementado con `scikit-fuzzy`.

Variables lingüísticas:
  - acoustic (bajo / medio / alto)
  - lyrics   (bajo / medio / alto)
  - content_ai (bajo / medio / alto / muy_alto)

Reglas (resumen):
  - Si acústica es alta Y letras es alta          -> content_ai muy_alto
  - Si acústica es alta Y letras es media          -> content_ai alto
  - Si acústica es media Y letras es alta          -> content_ai alto
  - Si acústica es alta Y letras es baja           -> content_ai medio
  - Si acústica es media Y letras es media         -> content_ai medio
  - Si acústica es baja Y letras es alta           -> content_ai medio
  - Si acústica es media Y letras es baja          -> content_ai bajo
  - Si acústica es baja Y letras es media          -> content_ai bajo
  - Si acústica es baja Y letras es baja           -> content_ai bajo

Si las letras no están disponibles, se usa una versión simplificada (solo
acústica) o un fallback ponderado si `scikit-fuzzy` no está instalado.
"""
from __future__ import annotations
from utils import clip, safe_round
from config import WEIGHTS

try:
    import numpy as np
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKFUZZY_AVAILABLE = False


def _build_control_system():
    universe = np.arange(0, 101, 1)

    acoustic = ctrl.Antecedent(universe, "acoustic")
    lyrics = ctrl.Antecedent(universe, "lyrics")
    content_ai = ctrl.Consequent(universe, "content_ai")

    acoustic.automf(3, names=["bajo", "medio", "alto"])
    lyrics.automf(3, names=["bajo", "medio", "alto"])

    content_ai["bajo"] = fuzz.trimf(universe, [0, 0, 40])
    content_ai["medio"] = fuzz.trimf(universe, [25, 45, 65])
    content_ai["alto"] = fuzz.trimf(universe, [50, 70, 90])
    content_ai["muy_alto"] = fuzz.trimf(universe, [75, 100, 100])

    rules = [
        ctrl.Rule(acoustic["alto"] & lyrics["alto"], content_ai["muy_alto"]),
        ctrl.Rule(acoustic["alto"] & lyrics["medio"], content_ai["alto"]),
        ctrl.Rule(acoustic["medio"] & lyrics["alto"], content_ai["alto"]),
        ctrl.Rule(acoustic["alto"] & lyrics["bajo"], content_ai["medio"]),
        ctrl.Rule(acoustic["medio"] & lyrics["medio"], content_ai["medio"]),
        ctrl.Rule(acoustic["bajo"] & lyrics["alto"], content_ai["medio"]),
        ctrl.Rule(acoustic["medio"] & lyrics["bajo"], content_ai["bajo"]),
        ctrl.Rule(acoustic["bajo"] & lyrics["medio"], content_ai["bajo"]),
        ctrl.Rule(acoustic["bajo"] & lyrics["bajo"], content_ai["bajo"]),
    ]
    return ctrl.ControlSystem(rules)


_CONTROL_SYSTEM = None


def _get_control_system():
    global _CONTROL_SYSTEM
    if _CONTROL_SYSTEM is None:
        _CONTROL_SYSTEM = _build_control_system()
    return _CONTROL_SYSTEM


def fuzzy_combine(acoustic_score: float, lyrics_score: float | None) -> dict:
    """
    Retorna dict: {"method", "content_ai_score", "inputs_used": {...}}
    """
    acoustic_score = clip(acoustic_score if acoustic_score is not None else 50)

    if lyrics_score is None:
        # Sin letras disponibles: el score de contenido depende solo de lo acústico.
        return {
            "method": "acoustic_only (sin letra disponible, fuzzy no aplicable)",
            "content_ai_score": safe_round(acoustic_score, 2),
            "inputs_used": {
                "acoustic_ai_likelihood": safe_round(acoustic_score, 2),
                "lyrics_ai_likelihood": None,
            },
        }

    lyrics_score = clip(lyrics_score)

    if not SKFUZZY_AVAILABLE:
        weighted = (
            acoustic_score * WEIGHTS["acoustic_weight"] +
            lyrics_score * WEIGHTS["lyrics_weight"]
        )
        return {
            "method": "weighted_average_fallback (scikit-fuzzy no disponible)",
            "content_ai_score": safe_round(clip(weighted), 2),
            "inputs_used": {
                "acoustic_ai_likelihood": safe_round(acoustic_score, 2),
                "lyrics_ai_likelihood": safe_round(lyrics_score, 2),
            },
        }

    try:
        system = _get_control_system()
        sim = ctrl.ControlSystemSimulation(system)
        sim.input["acoustic"] = acoustic_score
        sim.input["lyrics"] = lyrics_score
        sim.compute()
        content_score = float(sim.output["content_ai"])
        return {
            "method": "mamdani_fuzzy_inference (scikit-fuzzy)",
            "content_ai_score": safe_round(clip(content_score), 2),
            "inputs_used": {
                "acoustic_ai_likelihood": safe_round(acoustic_score, 2),
                "lyrics_ai_likelihood": safe_round(lyrics_score, 2),
            },
        }
    except Exception:
        # Fallback robusto si la simulación difusa falla por cualquier motivo.
        weighted = (
            acoustic_score * WEIGHTS["acoustic_weight"] +
            lyrics_score * WEIGHTS["lyrics_weight"]
        )
        return {
            "method": "weighted_average_fallback (error en simulación difusa)",
            "content_ai_score": safe_round(clip(weighted), 2),
            "inputs_used": {
                "acoustic_ai_likelihood": safe_round(acoustic_score, 2),
                "lyrics_ai_likelihood": safe_round(lyrics_score, 2),
            },
        }
