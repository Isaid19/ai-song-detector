"""
api.py
========
API REST (Flask) para exponer el detector de música generada por IA,
pensada para ser consumida desde una app React Native (o cualquier cliente
HTTP).

Endpoints
---------
GET  /api/v1/health
    Chequeo de salud del servicio.

POST /api/v1/analyze
    multipart/form-data:
      - audio_file  (archivo, requerido)  -> mp3/wav/flac/m4a/ogg...
      - lyrics      (texto, opcional)     -> letra de la canción, si se tiene
      - skip_shazam (opcional, "true"/"false") -> omite la consulta a Shazam
    Devuelve el JSON completo descrito en ai_detector.analyze_song().

Ejecutar localmente:
    pip install -r requirements.txt
    python api.py
El servidor arranca en http://0.0.0.0:5000

Ejemplo de consumo desde React Native (fetch + FormData):

    const form = new FormData();
    form.append('audio_file', {
      uri: fileUri,
      name: 'cancion.mp3',
      type: 'audio/mpeg',
    });
    form.append('lyrics', letraDeLaCancion); // opcional

    const res = await fetch('http://<TU_IP>:5000/api/v1/analyze', {
      method: 'POST',
      body: form,
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    const json = await res.json();
"""
from __future__ import annotations
import os
import tempfile
import traceback
import uuid

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from ai_detector import analyze_song

ALLOWED_EXTENSIONS = {"mp3", "wav", "flac", "m4a", "ogg", "aac", "wma", "aiff"}
MAX_CONTENT_LENGTH_MB = 60

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024
CORS(app)  # habilita llamadas desde apps móviles / front-ends externos


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/api/v1/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "ai-song-detector",
        "message": "El servicio está operativo.",
    })


@app.post("/api/v1/analyze")
def analyze():
    if "audio_file" not in request.files:
        return jsonify({
            "error": "No se envió ningún archivo en el campo 'audio_file'."
        }), 400

    file = request.files["audio_file"]
    if file.filename == "":
        return jsonify({"error": "El nombre del archivo está vacío."}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "error": f"Extensión no soportada. Formatos permitidos: {sorted(ALLOWED_EXTENSIONS)}"
        }), 400

    lyrics_text = request.form.get("lyrics")
    skip_shazam = str(request.form.get("skip_shazam", "false")).lower() in ("1", "true", "yes")

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, unique_name)

    try:
        file.save(tmp_path)
        result = analyze_song(tmp_path, lyrics_text=lyrics_text, skip_shazam=skip_shazam)
        # Se reemplaza el nombre de archivo temporal por el nombre original del usuario.
        result["audio_file"] = filename
        return jsonify(result), 200
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({
            "error": "Ocurrió un error al analizar el archivo de audio.",
            "detail": str(exc),
        }), 500
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.errorhandler(413)
def too_large(_e):
    return jsonify({
        "error": f"El archivo excede el tamaño máximo permitido ({MAX_CONTENT_LENGTH_MB} MB)."
    }), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
