"""
lyrics_analysis.py
=====================
Analiza el texto de la letra de una canción (cuando está disponible, ya sea
porque el usuario la envía en la petición o porque se obtuvo de otra fuente)
para estimar, mediante heurísticas ligeras basadas en NLP clásico (sin
dependencias pesadas de modelos de lenguaje), qué tan probable es que haya
sido generada por IA en base a:

  - Naturalidad de la letra
  - Originalidad temática
  - Coherencia emocional
  - Patrones de repetición
  - Especificidad narrativa
  - Variación estructural esperada

Cada sub-score va de 0 (muy humano) a 100 (muy probablemente IA).

Nota: al no incluir aquí un motor de transcripción de audio a texto (ASR),
si no se proporciona la letra explícitamente, `available=False` y el score
de letras no participa en la fusión final (ver fuzzy_engine.py).
"""
from __future__ import annotations
import re
from collections import Counter

from utils import clip, safe_round

# Frases/clichés que aparecen con frecuencia inusual en letras generadas
# por modelos de IA musicales (temáticas genéricas y muy repetidas entre sí).
CLICHE_PHRASES = [
    "under the stars", "dance in the moonlight", "neon lights", "chase the dawn",
    "electric fire", "shining bright", "through the night", "feel alive",
    "break the chains", "rise above", "heart on fire", "lost in the moment",
    "bajo las estrellas", "luz de luna", "corazon en llamas", "corazón en llamas",
    "toda la noche", "sentirme vivo", "romper las cadenas", "brillar como el sol",
    "luces de neon", "luces de neón",
]

# Palabras de relleno / etiquetas de estructura que a veces "se filtran" en
# letras generadas automáticamente (marcadores de sección sin limpiar).
STRUCTURE_LEAK_MARKERS = [
    "[verse", "[chorus", "[bridge", "[outro", "[intro", "[hook",
    "(verse", "(chorus", "(bridge", "(outro", "(intro", "(hook",
    "verso 1", "coro:", "estribillo:",
]

POSITIVE_WORDS = {"love", "amor", "happy", "feliz", "light", "luz", "hope", "esperanza",
                   "free", "libre", "shine", "brillar", "dream", "sueño", "joy", "alegria", "alegría"}
NEGATIVE_WORDS = {"pain", "dolor", "cry", "llorar", "dark", "oscuro", "lost", "perdido",
                   "broken", "roto", "fear", "miedo", "alone", "solo", "sad", "triste"}

CONCRETE_NOUN_HINTS = [
    r"\bcalle [A-ZÁÉÍÓÚ]\w+", r"\b\d{1,2}:\d{2}\b", r"\b19\d{2}\b", r"\b20\d{2}\b",
]


def _tokenize(text: str):
    return re.findall(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ']+", text.lower())


def _lines(text: str):
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _naturalness_score(lines, tokens, raw_text) -> float:
    score = 0.0
    total_signals = 0

    # 1) Fugas de marcadores de estructura sin limpiar (fuerte indicio de IA)
    leaks = sum(1 for marker in STRUCTURE_LEAK_MARKERS if marker in raw_text.lower())
    if leaks > 0:
        score += 90
    total_signals += 1

    # 2) Diversidad léxica (type-token ratio). Muy baja o extremadamente alta
    #    y "perfecta" puede ser señal de generación automática.
    if tokens:
        ttr = len(set(tokens)) / len(tokens)
        # rango humano típico ~0.35-0.65 en letras de canciones; fuera de rango, sospechoso.
        if ttr < 0.25 or ttr > 0.85:
            score += 65
        else:
            score += 25
    total_signals += 1

    # 3) Longitud de línea extremadamente uniforme (mecánica) => sospechoso
    if len(lines) >= 4:
        lengths = [len(l.split()) for l in lines]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        std = variance ** 0.5
        cv = std / mean_len if mean_len else 0
        score += 80 if cv < 0.12 else 30
        total_signals += 1

    return clip(score / total_signals if total_signals else 50)


def _thematic_originality_score(raw_text: str) -> float:
    text_l = raw_text.lower()
    hits = sum(1 for phrase in CLICHE_PHRASES if phrase in text_l)
    if hits == 0:
        return 20.0
    return clip(20 + hits * 18)


def _emotional_coherence_score(lines) -> float:
    if not lines:
        return 50.0
    polarities = []
    for line in lines:
        tokens = set(_tokenize(line))
        pos = len(tokens & POSITIVE_WORDS)
        neg = len(tokens & NEGATIVE_WORDS)
        if pos == 0 and neg == 0:
            continue
        polarities.append(1 if pos > neg else (-1 if neg > pos else 0))
    if len(polarities) < 2:
        return 45.0
    # Cuenta cuántas veces cambia bruscamente la polaridad emocional línea a línea.
    flips = sum(1 for i in range(1, len(polarities)) if polarities[i] != polarities[i - 1])
    flip_ratio = flips / (len(polarities) - 1)
    # Demasiados cambios erráticos (o ninguno, monotonía absoluta) sugiere falta
    # de arco emocional natural.
    if flip_ratio > 0.75 or flip_ratio == 0.0:
        return 70.0
    return 30.0


def _repetition_patterns_score(lines, tokens) -> float:
    if not lines:
        return 50.0
    line_counts = Counter(lines)
    most_common_count = line_counts.most_common(1)[0][1] if line_counts else 1
    repetition_ratio = most_common_count / len(lines)

    # n-gramas de 3 palabras repetidos en exceso
    trigrams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)] if len(tokens) >= 3 else []
    tri_counts = Counter(trigrams)
    repeated_trigrams = sum(1 for _, c in tri_counts.items() if c >= 3)

    score = 0.0
    # Repetición moderada de coro es normal (~0.2-0.4); excesiva (>0.55) es sospechosa.
    score += 75 if repetition_ratio > 0.55 else (20 if repetition_ratio < 0.15 else 35)
    score += min(40, repeated_trigrams * 10)
    return clip(score / 1.4)


def _narrative_specificity_score(raw_text: str, tokens) -> float:
    concrete_hits = 0
    for pattern in CONCRETE_NOUN_HINTS:
        concrete_hits += len(re.findall(pattern, raw_text))
    # Nombres propios (heurística simple: palabras capitalizadas no al inicio de línea)
    capitalized = re.findall(r"(?<!^)(?<!\.\s)\b[A-Z][a-záéíóúñ]{2,}\b", raw_text, flags=re.MULTILINE)
    specificity_signals = concrete_hits + len(set(capitalized))

    if not tokens:
        return 50.0
    density = specificity_signals / max(1, len(tokens) / 40)  # normalizado cada ~40 palabras
    if density < 0.3:
        return 75.0  # muy poca especificidad concreta -> más genérico/IA
    if density > 1.5:
        return 20.0
    return 45.0


def _structural_variation_score(lines) -> float:
    if len(lines) < 6:
        return 50.0
    # Agrupar en bloques de 4 líneas simulando estrofas y comparar longitudes.
    block_size = 4
    blocks = [lines[i:i + block_size] for i in range(0, len(lines), block_size)]
    block_word_counts = [sum(len(l.split()) for l in b) for b in blocks if b]
    if len(block_word_counts) < 2:
        return 50.0
    mean_wc = sum(block_word_counts) / len(block_word_counts)
    variance = sum((c - mean_wc) ** 2 for c in block_word_counts) / len(block_word_counts)
    std = variance ** 0.5
    cv = std / mean_wc if mean_wc else 0
    # Estructuras casi idénticas entre estrofas (muy baja variación) es indicio de plantilla IA.
    return clip(80 if cv < 0.08 else (25 if cv > 0.35 else 45))


def analyze_lyrics(lyrics_text: str | None, source: str = "user_provided") -> dict:
    """
    Punto de entrada del módulo. Si `lyrics_text` es None o vacío, retorna
    `available=False` y todos los sub-scores en None (no participan en la fusión).
    """
    if not lyrics_text or not lyrics_text.strip():
        return {
            "available": False,
            "word_count": 0,
            "line_count": 0,
            "naturalness_ai_score": None,
            "thematic_originality_ai_score": None,
            "emotional_coherence_ai_score": None,
            "repetition_patterns_ai_score": None,
            "narrative_specificity_ai_score": None,
            "structural_variation_ai_score": None,
            "lyrics_ai_likelihood": None,
            "source": None,
        }

    lines = _lines(lyrics_text)
    tokens = _tokenize(lyrics_text)

    naturalness = _naturalness_score(lines, tokens, lyrics_text)
    thematic = _thematic_originality_score(lyrics_text)
    emotional = _emotional_coherence_score(lines)
    repetition = _repetition_patterns_score(lines, tokens)
    specificity = _narrative_specificity_score(lyrics_text, tokens)
    structural = _structural_variation_score(lines)

    weights = {
        "naturalness": 0.22, "thematic": 0.18, "emotional": 0.15,
        "repetition": 0.17, "specificity": 0.15, "structural": 0.13,
    }
    lyrics_ai_likelihood = (
        naturalness * weights["naturalness"] +
        thematic * weights["thematic"] +
        emotional * weights["emotional"] +
        repetition * weights["repetition"] +
        specificity * weights["specificity"] +
        structural * weights["structural"]
    )

    return {
        "available": True,
        "word_count": len(tokens),
        "line_count": len(lines),
        "naturalness_ai_score": safe_round(naturalness, 2),
        "thematic_originality_ai_score": safe_round(thematic, 2),
        "emotional_coherence_ai_score": safe_round(emotional, 2),
        "repetition_patterns_ai_score": safe_round(repetition, 2),
        "narrative_specificity_ai_score": safe_round(specificity, 2),
        "structural_variation_ai_score": safe_round(structural, 2),
        "lyrics_ai_likelihood": safe_round(clip(lyrics_ai_likelihood), 2),
        "source": source,
    }
