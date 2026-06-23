# 🛡️ ToxiWatch — Multilingual Toxicity Moderation System

> **Automatically detects and moderates toxic comments in English, Hindi, and Hinglish using BERT-based deep learning.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-red?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Features

- 🌐 **Multilingual** — English, Hindi (Devanagari script), and Hinglish (romanized Hindi)
- 🤖 **BERT-based toxicity scoring** via [Detoxify](https://github.com/unitaryai/detoxify)
- 🔄 **Auto-translation** — Hindi/Hinglish → English via [Helsinki-NLP/opus-mt-hi-en](https://huggingface.co/Helsinki-NLP/opus-mt-hi-en)
- 🏷️ **3-tier moderation** — `APPROVED` / `REVIEW` / `BLOCKED`
- 📊 **Live dashboard** — Real-time history, stats, and toxicity score visualization
- 🗄️ **SQLite persistence** — All moderation results stored with full audit trail
- 📚 **Auto API docs** — Interactive Swagger UI at `/docs`

---

## 🖼️ Dashboard

The dashboard provides:
- Comment submission with quick test buttons (English / Hindi / Hinglish samples)
- Animated toxicity gauge and category score bars
- Moderation history table with language, score, and decision
- Live stats: total moderated, approved, reviewed, blocked

---

## 🏗️ Architecture

```
ToxiWatch
├── main.py                    # FastAPI entry point + lifespan (model loading)
├── app/
│   ├── routes.py              # API endpoints: /moderate /history /stats /test-comments
│   ├── ml_engine.py           # Language detection + translation + toxicity scoring
│   ├── database.py            # SQLite CRUD operations
│   └── models.py              # Pydantic v2 request/response schemas
├── static/
│   └── index.html             # Frontend dashboard (vanilla HTML/CSS/JS)
├── tests/
│   └── test_moderation.py     # 30 pytest test cases
├── requirements.txt           # Python dependencies
├── setup_windows.bat          # One-click Windows setup
└── start_server.bat           # One-click server start
```

### ML Pipeline

```
User Comment
     │
     ▼
Language Detection (langdetect + Devanagari regex + Hinglish keywords)
     │
     ├── English  ──────────────────────────────┐
     ├── Hindi (Devanagari) ──→ Translation ────┤
     └── Hinglish (Roman)   ──→ Translation ────┘
                                                 │
                                                 ▼
                                    Detoxify BERT Scoring
                                    (toxicity, insult, threat,
                                     obscene, identity_attack...)
                                                 │
                                                 ▼
                              Score < 0.4  →  APPROVED
                              Score 0.4–0.7 → REVIEW
                              Score ≥ 0.7  →  BLOCKED
```

---

## 🚀 Quick Start (Windows)

### Prerequisites
- **Python 3.11** — [Download](https://python.org/downloads/) (check "Add to PATH")
- **Git** — [Download](https://git-scm.com/download/win)
- ~1.5 GB free disk space (for ML model weights, downloaded once on first run)

### Step 1 — Clone the repository
```bash
git clone https://github.com/Sanxkar/toxiwatch.git
cd toxiwatch
```

### Step 2 — Run the setup script
```bat
setup_windows.bat
```
This will:
- Create a Python 3.11 virtual environment
- Install all dependencies (including PyTorch CPU-only, ~500MB)
- Verify all imports work

### Step 3 — Start the server
```bat
start_server.bat
```
Or manually:
```powershell
$env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"
.\venv\Scripts\uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

> **⚠️ First startup:** Downloads ML model weights (~1 GB). This is a **one-time** download. Subsequent starts take ~30 seconds.

### Step 4 — Open the dashboard
Navigate to **http://localhost:8000**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Frontend dashboard |
| `GET` | `/health` | Health check — `{"status": "ok"}` |
| `POST` | `/moderate` | Analyze a comment for toxicity |
| `GET` | `/history` | Paginated moderation history |
| `GET` | `/stats` | Summary statistics |
| `GET` | `/test-comments` | 30 pre-built test samples |
| `GET` | `/docs` | Interactive Swagger API documentation |

### POST /moderate — Request
```json
{
  "comment": "Tu bahut bada bewakoof hai yaar"
}
```

### POST /moderate — Response
```json
{
  "original_comment": "Tu bahut bada bewakoof hai yaar",
  "translated_comment": "You are a very big fool, friend",
  "detected_language": "hinglish",
  "text_analyzed": "You are a very big fool, friend",
  "toxicity_score": 0.7823,
  "scores": {
    "toxicity": 0.7823,
    "severe_toxicity": 0.0412,
    "obscene": 0.0231,
    "threat": 0.0089,
    "insult": 0.6541,
    "identity_attack": 0.0103
  },
  "action": "BLOCKED",
  "log_id": 42
}
```

---

## 🧪 Running Tests

```powershell
.\venv\Scripts\pytest tests\test_moderation.py -v
```

The test suite includes 30 test cases covering:
- ✅ English safe comments → `APPROVED`
- ⚠️ English mildly toxic → `REVIEW`
- 🚫 English severely toxic → `BLOCKED`
- 🇮🇳 Hindi Devanagari (safe + toxic)
- 🇮🇳 Hinglish (safe + mild + severe)
- 🔀 Mixed language edge cases

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| FastAPI | 0.111.0 | Web framework |
| Uvicorn | 0.29.0 | ASGI server |
| Pydantic | 2.7.1 | Data validation (v2 API) |
| PyTorch | 2.3.1+cpu | Deep learning runtime |
| Detoxify | 0.5.2 | BERT toxicity classifier |
| Transformers | 4.41.2 | HuggingFace model loading |
| SentencePiece | 0.1.99 | MarianMT tokenizer |
| langdetect | 1.0.9 | Language detection |

---

## 🗂️ Moderation Thresholds

| Score Range | Decision | Meaning |
|-------------|----------|---------|
| `0.00 – 0.39` | ✅ `APPROVED` | Safe to publish |
| `0.40 – 0.69` | ⚠️ `REVIEW` | Needs human review |
| `0.70 – 1.00` | 🚫 `BLOCKED` | Automatically rejected |

---

## 🪟 Windows-Specific Notes

- The setup script sets `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` to prevent `UnicodeEncodeError` on Windows consoles.
- SQLite database is created at `toxiwatch.db` in the project root on first startup.
- ML model weights are cached in `C:\Users\<you>\.cache\` and reused on subsequent starts.
- Run `setup_windows.bat` from **Command Prompt** (not PowerShell) for best compatibility.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
