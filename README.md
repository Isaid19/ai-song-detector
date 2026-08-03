# 🎵 AI Song Detector — Backend (Python)

Sistema en Python que analiza una canción y estima, en una escala de **0
(Humano) a 100 (Generado por IA)**, la probabilidad de que haya sido creada
con una herramienta de inteligencia artificial (Suno, Gemini, Mureka, AIVA,
Mubert, Soundful, Soundraw, LANDR, Beatoven, Moises, Undio, MusicGPT,
ElevenLabs, Loudly, Amper Music, Canva, etc.).

## 🧠 Lógica de decisión (en capas)

1. **Metadatos** (`metadata_analysis.py`): se leen los tags del archivo
   (título, artista, álbum, año, género, comentario, encoder...) usando
   `mutagen`, buscando firmas de herramientas de IA conocidas. Si se
   encuentra una, el puntaje de IA sube fuertemente (score ~97/100).
2. **ShazamIO** (`shazam_analysis.py`): si los metadatos no son
   concluyentes, se intenta reconocer la canción por huella acústica. Si
   Shazam la reconoce (ya es una grabación existente), el puntaje de IA baja
   considerablemente (evidencia de origen humano).
3. **Análisis acústico + letra con lógica difusa** (`acoustic_analysis.py`,
   `lyrics_analysis.py`, `fuzzy_engine.py`): si ninguna de las anteriores es
   concluyente, se extraen características con `librosa`
   (ZCR, centroide/ancho de banda/contraste/rolloff/flujo espectral, MFCC,
   RMSE) y —si se dispone del texto— se evalúan heurísticas de la letra
   (naturalidad, originalidad temática, coherencia emocional, repetición,
   especificidad narrativa, variación estructural). Ambas señales se
   combinan con un sistema de inferencia difusa (Mamdani) usando
   `scikit-fuzzy`.

El resultado siempre combina, en distinta proporción según la capa
dominante, las tres fuentes de evidencia (ver `config.WEIGHTS`).

> `pyAudioAnalysis` se integra como capa **opcional** de validación cruzada
> (si está instalada); el sistema funciona igual de bien solo con `librosa`.

## 📦 Instalación

```bash
cd backend
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Uso por línea de comandos

```bash
python cli.py cancion.mp3
python cli.py cancion.mp3 --lyrics letra.txt
python cli.py cancion.mp3 --skip-shazam
```

## 🌐 Levantar la API REST

```bash
python api.py
# Servidor disponible en http://0.0.0.0:5000
```

### Endpoints

- `GET /api/v1/health` — chequeo de salud.
- `POST /api/v1/analyze` — analiza un archivo de audio.
  - `multipart/form-data`:
    - `audio_file` (**requerido**): archivo `.mp3/.wav/.flac/.m4a/.ogg/...`
    - `lyrics` (opcional): texto de la letra de la canción.
    - `skip_shazam` (opcional): `"true"` para omitir la consulta a Shazam.

#### Ejemplo con `curl`

```bash
curl -X POST http://localhost:5000/api/v1/analyze \
  -F "audio_file=@/ruta/a/cancion.mp3" \
  -F "lyrics=@/ruta/a/letra.txt;type=text/plain"
```

#### Ejemplo desde **React Native**

```js
const form = new FormData();
form.append('audio_file', {
  uri: fileUri,          // uri local del archivo (expo-document-picker, etc.)
  name: 'cancion.mp3',
  type: 'audio/mpeg',
});
form.append('lyrics', letraDeLaCancion); // opcional

const res = await fetch('http://TU_IP_O_DOMINIO:5000/api/v1/analyze', {
  method: 'POST',
  body: form,
  headers: { 'Content-Type': 'multipart/form-data' },
});
const json = await res.json();
console.log(json.ai_score, json.classification);
```

## 📄 Formato de respuesta (resumen)

```jsonc
{
  "audio_file": "cancion.mp3",
  "scale": { "0": "Humano", "100": "Generado por IA" },
  "ai_score": 82.4,
  "ai_score_bar": "[████████████████░░░░] 82%",
  "classification": "Probablemente generado por IA",
  "decision_layer": "metadata | shazam | acoustic_lyrics_fuzzy",
  "decision_reasoning": "...",
  "shazam_analysis": { "skipped": false, "recognized": true, "...": "..." },
  "metadata_analysis": { "ai_tool_detected": "Suno", "...": "..." },
  "acoustic_and_lyrics_analysis": {
    "acoustic_features": { "zcr": {"...":"..."}, "mfcc": {"...":"..."} },
    "lyrics_analysis": { "available": false },
    "fuzzy_inference": { "method": "mamdani_fuzzy_inference (scikit-fuzzy)", "content_ai_score": 74.1 }
  },
  "Zero Crossing Rate": { "mean": 0.08, "ai_flatness_score": 61.2, "bar": "[████████████░░░░░░░░] 61%" },
  "Spectral Centroid": { "...": "..." },
  "Spectral Bandwidth": { "...": "..." },
  "Spectral Contrast": { "...": "..." },
  "Spectral Rolloff": { "...": "..." },
  "Spectral Flux": { "...": "..." },
  "MFCCs": { "...": "..." },
  "RMSE": { "...": "..." }
}
```

## ⚠️ Limitaciones

- El análisis acústico/lírico es **heurístico**: útil como capa de apoyo
  cuando no hay evidencia directa (metadatos o reconocimiento de la pista),
  pero no es un clasificador entrenado con machine learning sobre un dataset
  etiquetado. Se recomienda complementarlo a futuro con un modelo supervisado.
- El reconocimiento por Shazam requiere conexión a internet.
- La extracción de letra (ASR) no está incluida; si se desea analizar la
  letra automáticamente a partir del propio audio, se puede integrar
  `openai-whisper` o `faster-whisper` y pasar el texto resultante a
  `lyrics_analysis.analyze_lyrics()`.
