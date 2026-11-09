# Uncensored Talking (@uncensored_talking) — Automated Short-Form Clipper

Vollautomatisches CLI-Pipeline-Tool zur Extraktion viraler 9:16 Shorts aus YouTube-Videos oder lokalen Videoaufnahmen (z. B. The Joe Rogan Experience, Hot Ones, Tucker Carlson, Lex Fridman).

Mit **automatischem YouTube-Download**, **B-Roll-Recherche & Ken-Burns-Motion**, **dynamischem OpenCV Face-Tracking** und **3 automatischen Layout-Modi**.

---

## ⚡ Schnellstart

### 1. Installation

```bash
# Virtuelle Umgebung aktivieren (oder erstellen)
source .venv/bin/activate

# Abhängigkeiten installieren
uv pip install -r requirements.txt
```

### 2. Konfiguration (optional)

In `.env` können API-Keys für LLM und B-Roll-Dienste hinterlegt werden (falls nicht vorhanden, greift die Pipeline vollautomatisch auf integrierte Offline-Heuristiken & lizenzfreie Assets zurück):

```env
# LLM für Schnitt- & Szenenregie
OPENAI_API_KEY=dein_api_key
LLM_MODEL=gpt-4o

# Kostenlose B-Roll APIs (optional)
PEXELS_API_KEY=dein_pexels_api_key
PIXABAY_API_KEY=dein_pixabay_api_key

# Hardware & Encoder
WHISPER_MODEL=large-v3
WHISPER_DEVICE=auto
VIDEO_ENCODER=auto
```

### 3. Pipeline per YouTube-Link starten

```bash
python main.py --url "https://www.youtube.com/watch?v=IH_q5r2bLCE"
```

Oder mit lokaler Datei:

```bash
python main.py --input input/episode_1.mp4
```

Die fertigen vertikalen Shorts landen in `output/` (z. B. `output/clip_1_uncensored_clip_1.mp4`).

---

## 🎨 3 Dynamische Layout-Modi

Für jeden Clip erstellt die Regie-Engine alle 2 bis 5 Sekunden automatische Szenenwechsel:
1. **`full_speaker`**: 1080x1920 Sprecher im Hochformat mit OpenCV Face-Centering.
2. **`full_broll`**: 1080x1920 Bildgewaltiges B-Roll Footage oder hochauflösendes Bild mit stetigem Ken-Burns Zoom/Pan (1.0x -> 1.15x).
3. **`splitscreen`**:
   - **Oben (1080x960)**: Thematisches B-Roll Material
   - **Unten (1080x960)**: Sprecher zentriert auf Gesicht
   - **Zentrum**: Untertitel sitzen exakt auf der Schnittlinie (`Alignment: 5`)

---

## 📁 Ordnerstruktur

```text
.
├── input/             # YouTube-Downloads & lokale MP4s
├── output/            # Fertige 9:16 Shorts mit eingebrannten Untertiteln
├── temp/              # Audio-Dumps, Transkripte, B-Roll Assets, Szenenplatten, ASS-Dateien
├── src/
│   ├── __init__.py
│   ├── downloader.py  # Automatischer YouTube-Download via yt-dlp (1080p MP4)
│   ├── audio.py       # Extraktion via FFmpeg (16kHz mono WAV)
│   ├── transcribe.py  # faster-whisper large-v3: TXT + Word-Level Timestamps JSON
│   ├── selector.py    # LLM & Heuristik: 30-60s Schnitt, 3s-Hook & 2-5s Szenenplan
│   ├── broll_engine.py# Sourcing (Pexels, Pixabay, CC0) + Ken-Burns Motion-Engine
│   ├── subtitles.py   # ASS-Subtitles (Center Alignment 5, Gold & dicker Rand, Hold-Logik)
│   └── video.py       # OpenCV Face-Tracking, 3 Layout-Modi & NVENC/libx264 Rendering
├── tests/             # Vollständige Test-Suite (13 automatisierte Tests)
├── main.py            # CLI-Einstiegspunkt
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🛠️ CLI-Optionen

```bash
python main.py [OPTIONEN]

Optionen:
  -u, --url URL         YouTube-URL zum automatischen Herunterladen und Verarbeiten
  -i, --input PATH      Pfad zur bestehenden Eingabedatei (z. B. input/podcast.mp4)
  -c, --clips N         Anzahl der zu generierenden Clips (Standard: 3)
  -m, --model MODEL     Whisper-Modell (Standard: large-v3, Alternativen: medium, small, base)
  -d, --device DEVICE   Hardware für Whisper: 'auto', 'cuda', 'cpu' (Standard: auto)
  --compute-type TYPE   Berechnungstyp: 'auto', 'float16', 'int8', 'float32'
  --encoder ENCODER     Video-Encoder: 'auto', 'h264_nvenc', 'libx264'
  --skip-transcribe     Bestehende Transkripte wiederverwenden (spart Zeit bei Re-Runs)
  --offline             Erzwingt Offline-Heuristik (kein LLM-API-Call)
  --font FONT           Schriftart für ASS-Untertitel (Standard: Montserrat-Black)
  --color HEX           Primärfarbe der Untertitel in ASS BGR (Standard: &H002BF7FF für Gold/Gelb)
  --force               Erzwingt Neu-Download und Neu-Generierung aller Schritte
  -v, --verbose         Ausführliche Debug-Logs
```

---

## 🧪 Tests ausführen

```bash
PYTHONPATH=. .venv/bin/pytest tests/
```
