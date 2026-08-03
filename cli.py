"""
cli.py
========
Utilidad de línea de comandos para probar el detector sin necesidad de
levantar el servidor Flask.

Uso:
    python cli.py ruta/a/cancion.mp3
    python cli.py ruta/a/cancion.mp3 --lyrics ruta/a/letra.txt
    python cli.py ruta/a/cancion.mp3 --skip-shazam
"""
import argparse
import json
import sys

from ai_detector import analyze_song


def main():
    parser = argparse.ArgumentParser(description="Detector de canciones generadas por IA.")
    parser.add_argument("audio_path", help="Ruta al archivo de audio a analizar.")
    parser.add_argument("--lyrics", help="Ruta a un archivo .txt con la letra de la canción.", default=None)
    parser.add_argument("--skip-shazam", action="store_true", help="Omite la consulta al servicio de Shazam.")
    args = parser.parse_args()

    lyrics_text = None
    if args.lyrics:
        with open(args.lyrics, "r", encoding="utf-8") as f:
            lyrics_text = f.read()

    result = analyze_song(args.audio_path, lyrics_text=lyrics_text, skip_shazam=args.skip_shazam)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
