"""
acoustic_analysis.py
======================
Extrae características acústicas de bajo nivel con `librosa` (y, si está
disponible, complementa con `pyAudioAnalysis`) para estimar qué tan "natural"
(humana) o "artificial" (IA) suena una pista, en ausencia de evidencia
concluyente de metadatos o de Shazam.

Características calculadas:
  - Zero Crossing Rate (ZCR)
  - Spectral Centroid
  - Spectral Bandwidth
  - Spectral Contrast
  - Spectral Rolloff
  - Spectral Flux
  - MFCCs (timbre / envolvente espectral)
  - RMSE (energía)

Heurística utilizada
---------------------
La música generada por modelos de IA actuales tiende a producir texturas
tímbricas y dinámicas *demasiado regulares/planas* frame a frame (menor
variabilidad natural que la que introduce una grabación humana real con
micrófono, instrumentos acústicos, respiración, dinámica interpretativa,
ruido de sala, etc.). Por eso, para la mayoría de las características se
usa el "coeficiente de variación" (std/mean) como proxy de naturalidad:
  - CV bajo  -> señal muy uniforme  -> mayor "ai_flatness_score"
  - CV alto  -> señal muy variable  -> menor "ai_flatness_score"

Para los MFCC, en cambio, se evalúa la variabilidad general de los
coeficientes entre tramas (menor variabilidad tímbrica sostenida también
es un indicio de síntesis).

⚠️ Importante: esto es una heurística de apoyo, no un detector infalible.
Se documenta explícitamente en el campo "reasoning"/"acoustic_ai_likelihood".
"""
from __future__ import annotations
import numpy as np

from config import MAX_ANALYSIS_DURATION_SECONDS
from utils import stats_block, sigmoid_scale, clip, safe_round, make_bar

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:  # pragma: no cover
    LIBROSA_AVAILABLE = False

try:
    from pyAudioAnalysis import ShortTermFeatures as _paa_stf
    PYAUDIOANALYSIS_AVAILABLE = True
except Exception:  # pragma: no cover - dependencia pesada/opcional
    PYAUDIOANALYSIS_AVAILABLE = False


def _spectral_flux(S: np.ndarray) -> np.ndarray:
    """
    Calcula el flujo espectral: la variación de magnitud del espectro entre
    tramas consecutivas (raíz del error cuadrático medio de la diferencia).
    S: matriz de magnitudes espectrales (freq_bins x frames)
    """
    if S.shape[1] < 2:
        return np.zeros(1)
    diff = np.diff(S, axis=1)
    diff = np.clip(diff, a_min=0, a_max=None)  # solo incrementos de energía (flux "half-wave rectified")
    flux = np.sqrt(np.mean(diff ** 2, axis=0))
    return flux


def _feature_block(arr: np.ndarray, flatness_midpoint: float, flatness_steepness: float) -> dict:
    """
    Construye un bloque de estadísticas + un "ai_flatness_score" (0-100)
    a partir del coeficiente de variación de la característica.
    Un CV bajo (poca variación) implica mayor score de "planitud" -> más IA.
    """
    stats = stats_block(arr)
    cv = stats.get("coefficient_of_variation") or 0.0
    # A menor CV, mayor sospecha de IA. Se invierte con una sigmoide descendente.
    flatness = sigmoid_scale(-cv, midpoint=-flatness_midpoint, steepness=flatness_steepness)
    stats["ai_flatness_score"] = safe_round(flatness, 2)
    return stats


def _mfcc_block(mfcc: np.ndarray) -> dict:
    """
    mfcc: matriz (n_mfcc x frames).
    mean_of_means / mean_of_stds: describen el timbre promedio y su dispersión.
    overall_std: dispersión global de todos los coeficientes (variabilidad tímbrica).
    ai_variability_score: score 0-100 (mayor = más plano/artificial).
    """
    means_per_coef = np.mean(mfcc, axis=1)
    stds_per_coef = np.std(mfcc, axis=1)
    overall_std = float(np.std(mfcc))

    mean_of_means = float(np.mean(means_per_coef))
    mean_of_stds = float(np.mean(stds_per_coef))

    # Referencia empírica aproximada: en música/voz humana overall_std suele
    # rondar >= 12-18; valores notablemente menores sugieren timbre "plano".
    variability_score = sigmoid_scale(-overall_std, midpoint=-14.0, steepness=0.28)

    return {
        "mean_of_means": safe_round(mean_of_means, 4),
        "mean_of_stds": safe_round(mean_of_stds, 4),
        "overall_std": safe_round(overall_std, 4),
        "ai_variability_score": safe_round(variability_score, 2),
    }


def _try_pyaudioanalysis_supplement(y: np.ndarray, sr: int) -> dict:
    """
    Extracción complementaria usando pyAudioAnalysis (si está disponible).
    Se usa únicamente como validación cruzada de ZCR/energía/centroid, ya
    que pyAudioAnalysis calcula 34 features de corto plazo de forma nativa.
    Si la librería no está instalada o falla, se retorna un dict vacío y el
    resto del análisis continúa normalmente basado en librosa.
    """
    if not PYAUDIOANALYSIS_AVAILABLE:
        return {"available": False, "reason": "pyAudioAnalysis no está instalado (uso opcional)."}
    try:
        win = 0.050 * sr
        step = 0.025 * sr
        features, feature_names = _paa_stf.feature_extraction(y, sr, win, step)
        # feature_names típicamente: zcr, energy, energy_entropy, spectral_centroid,
        # spectral_spread, spectral_entropy, spectral_flux, spectral_rolloff, mfcc_1..13, ...
        idx = {name: i for i, name in enumerate(feature_names)}
        summary = {}
        for key in ("zcr", "energy", "spectral_centroid", "spectral_flux", "spectral_rolloff"):
            if key in idx:
                summary[key] = safe_round(float(np.mean(features[idx[key]])), 5)
        return {"available": True, "cross_validation_means": summary}
    except Exception as exc:
        return {"available": False, "reason": f"Fallo al ejecutar pyAudioAnalysis: {exc}"}


def analyze_acoustic(file_path: str, max_duration: int = MAX_ANALYSIS_DURATION_SECONDS) -> dict:
    """
    Punto de entrada del módulo. Devuelve un diccionario con:
      - "acoustic_features": bloque detallado (para acoustic_and_lyrics_analysis)
      - "summary_blocks": bloques resumidos por característica (mean/ai_flatness_score/bar)
        usados para las secciones de nivel superior del JSON final.
      - "acoustic_ai_likelihood": score global 0-100.
    """
    if not LIBROSA_AVAILABLE:
        empty_summary = {"mean": None, "ai_flatness_score": None, "bar": make_bar(0)}
        return {
            "acoustic_features": {
                "duration_analyzed_seconds": None,
                "sample_rate": None,
                "error": "librosa no está instalado en el servidor.",
            },
            "summary_blocks": {k: empty_summary for k in
                                ["Zero Crossing Rate", "Spectral Centroid", "Spectral Bandwidth",
                                 "Spectral Contrast", "Spectral Rolloff", "Spectral Flux",
                                 "MFCCs", "RMSE"]},
            "acoustic_ai_likelihood": 0.0,
        }

    y, sr = librosa.load(file_path, sr=None, mono=True, duration=max_duration)
    duration_analyzed = float(len(y) / sr) if sr else None

    # --- Features base ---
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rmse = librosa.feature.rms(y=y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    S = np.abs(librosa.stft(y))
    spectral_flux = _spectral_flux(S)

    # --- Bloques detallados (para acoustic_and_lyrics_analysis.acoustic_features) ---
    zcr_block = _feature_block(zcr, flatness_midpoint=0.35, flatness_steepness=8.0)
    centroid_block = _feature_block(spectral_centroid, flatness_midpoint=0.18, flatness_steepness=10.0)
    bandwidth_block = _feature_block(spectral_bandwidth, flatness_midpoint=0.15, flatness_steepness=10.0)
    contrast_block = _feature_block(spectral_contrast.flatten(), flatness_midpoint=0.35, flatness_steepness=8.0)
    rolloff_block = _feature_block(spectral_rolloff, flatness_midpoint=0.25, flatness_steepness=9.0)
    flux_block = _feature_block(spectral_flux, flatness_midpoint=0.45, flatness_steepness=7.0)
    rmse_block = _feature_block(rmse, flatness_midpoint=0.30, flatness_steepness=8.0)
    mfcc_block = _mfcc_block(mfcc)

    pyaa_supplement = _try_pyaudioanalysis_supplement(y, sr)

    # --- Score global acústico (promedio ponderado de los flatness/variability scores) ---
    component_scores = {
        "zcr": zcr_block["ai_flatness_score"],
        "spectral_centroid": centroid_block["ai_flatness_score"],
        "spectral_bandwidth": bandwidth_block["ai_flatness_score"],
        "spectral_contrast": contrast_block["ai_flatness_score"],
        "spectral_rolloff": rolloff_block["ai_flatness_score"],
        "spectral_flux": flux_block["ai_flatness_score"],
        "mfcc": mfcc_block["ai_variability_score"],
        "rmse": rmse_block["ai_flatness_score"],
    }
    weights = {
        "zcr": 0.10, "spectral_centroid": 0.12, "spectral_bandwidth": 0.10,
        "spectral_contrast": 0.10, "spectral_rolloff": 0.10, "spectral_flux": 0.13,
        "mfcc": 0.25, "rmse": 0.10,
    }
    valid_items = [(k, v) for k, v in component_scores.items() if v is not None]
    if valid_items:
        total_weight = sum(weights[k] for k, _ in valid_items)
        acoustic_ai_likelihood = sum(weights[k] * v for k, v in valid_items) / total_weight
    else:
        acoustic_ai_likelihood = 50.0

    acoustic_features = {
        "duration_analyzed_seconds": safe_round(duration_analyzed, 2),
        "sample_rate": int(sr) if sr else None,
        "zcr": zcr_block,
        "spectral_centroid": centroid_block,
        "mfcc": mfcc_block,
        "rmse": rmse_block,
        "pyaudioanalysis_cross_validation": pyaa_supplement,
        "acoustic_ai_likelihood": safe_round(clip(acoustic_ai_likelihood), 2),
    }

    def _summary(block, mean_key="mean", score_key="ai_flatness_score"):
        mean_val = block.get(mean_key)
        score_val = block.get(score_key)
        return {
            "mean": mean_val,
            "ai_flatness_score": score_val,
            "bar": make_bar(score_val or 0),
        }

    summary_blocks = {
        "Zero Crossing Rate": _summary(zcr_block),
        "Spectral Centroid": _summary(centroid_block),
        "Spectral Bandwidth": _summary(bandwidth_block),
        "Spectral Contrast": _summary(contrast_block),
        "Spectral Rolloff": _summary(rolloff_block),
        "Spectral Flux": _summary(flux_block),
        "MFCCs": {
            "mean": mfcc_block["mean_of_means"],
            "ai_flatness_score": mfcc_block["ai_variability_score"],
            "bar": make_bar(mfcc_block["ai_variability_score"] or 0),
        },
        "RMSE": _summary(rmse_block),
    }

    return {
        "acoustic_features": acoustic_features,
        "summary_blocks": summary_blocks,
        "acoustic_ai_likelihood": safe_round(clip(acoustic_ai_likelihood), 2),
    }
