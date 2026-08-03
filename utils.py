"""
utils.py
========
Funciones auxiliares compartidas entre los distintos módulos de análisis.
"""
from __future__ import annotations
import numpy as np


def clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Restringe un valor numérico al rango [lo, hi]."""
    try:
        return float(max(lo, min(hi, value)))
    except (TypeError, ValueError):
        return lo


def safe_round(value, decimals: int = 3):
    """Redondea de forma segura, devolviendo None si el valor no es válido."""
    try:
        if value is None:
            return None
        if isinstance(value, (int, float, np.floating, np.integer)):
            if np.isnan(value) or np.isinf(value):
                return None
            return round(float(value), decimals)
        return value
    except Exception:
        return None


def coefficient_of_variation(mean: float, std: float) -> float:
    """CV = std / |mean|. Devuelve 0 si la media es ~0 para evitar división por cero."""
    if mean is None or std is None:
        return 0.0
    if abs(mean) < 1e-9:
        return 0.0
    return float(std / abs(mean))


def make_bar(score: float, length: int = 20, filled_char: str = "█", empty_char: str = "░") -> str:
    """
    Genera una barra de progreso textual para representar visualmente un score 0-100.
    Ejemplo: make_bar(65) -> "[█████████████░░░░░░░] 65%"
    """
    score = clip(score, 0, 100)
    filled = int(round((score / 100.0) * length))
    filled = max(0, min(length, filled))
    bar = filled_char * filled + empty_char * (length - filled)
    return f"[{bar}] {round(score)}%"


def sigmoid_scale(x: float, midpoint: float, steepness: float) -> float:
    """
    Convierte un valor x en un score 0-100 usando una curva sigmoide,
    centrada en 'midpoint' con pendiente 'steepness'.
    Útil para transformar métricas acústicas crudas (p.ej. coeficiente de
    variación) en un "score de sospecha de IA" acotado y suave.
    """
    try:
        value = 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))
        return float(value * 100.0)
    except Exception:
        return 50.0


def stats_block(arr: np.ndarray) -> dict:
    """Calcula mean/std/min/max/CV de un array 1D de numpy."""
    arr = np.asarray(arr, dtype=float).flatten()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "coefficient_of_variation": None}
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "mean": safe_round(mean, 5),
        "std": safe_round(std, 5),
        "min": safe_round(float(np.min(arr)), 5),
        "max": safe_round(float(np.max(arr)), 5),
        "coefficient_of_variation": safe_round(coefficient_of_variation(mean, std), 5),
    }
