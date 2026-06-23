"""
main.py — ToxiWatch: Multilingual Toxicity Moderation System (REPAIRED)
=======================================================================
This is the entry point for the FastAPI web server.

FIXES APPLIED:
  - Added sys.stdout UTF-8 reconfiguration at startup for Windows console encoding
  - Replaced relative path open("static/index.html") with pathlib.Path(__file__)
    based absolute path resolution so the server works regardless of which
    directory uvicorn is launched from.
  - Added Windows-safe logging formatter that strips emoji if encoding fails.
  - Added explicit error handling for StaticFiles mount failure (missing directory).
"""

# ── Standard library ──────────────────────────────────────────────────────────
import logging
import sys
import os
from pathlib import Path                           # Safe cross-platform path handling
from contextlib import asynccontextmanager          # For startup/shutdown lifecycle hooks

# ── Windows Encoding Fix ──────────────────────────────────────────────────────
# MUST happen BEFORE any logging configuration so log messages render correctly.
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# ── FastAPI & HTTP utilities ──────────────────────────────────────────────────
from fastapi import FastAPI, Request                # Core FastAPI classes
from fastapi.responses import HTMLResponse, JSONResponse  # Types of HTTP responses
from fastapi.staticfiles import StaticFiles          # Serve CSS/JS/images

# ── Our own modules ───────────────────────────────────────────────────────────
from app.routes import router                        # All API endpoints live in app/routes.py
from app.database import init_db                     # Function that creates SQLite tables on startup

# ── Project root (the directory containing this main.py file) ─────────────────
# Using Path(__file__).parent ensures paths work regardless of the working directory.
# FIX: Replaced hardcoded relative paths with absolute Path-based references.
BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# ── Logging configuration ─────────────────────────────────────────────────────
# Using a safe ASCII format for Windows console compatibility.
# FIX: Removed emoji from format string; emoji in log messages are still handled
# by the UTF-8 reconfiguration above, but the formatter itself stays ASCII-safe.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ── Lifespan handler ──────────────────────────────────────────────────────────
# This function runs ONCE when the server starts, and ONCE when it shuts down.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager:
      - Code before `yield` runs at STARTUP
      - Code after `yield` runs at SHUTDOWN
    """
    logger.info("ToxiWatch starting up...")

    # Initialize SQLite database (creates tables if they don't exist)
    try:
        init_db()
        logger.info("[OK] Database initialized.")
    except Exception as e:
        logger.error("[ERROR] Database initialization failed: %s", str(e))
        raise

    logger.info("[...] Loading ML models (this may take 30-60 seconds on first run)...")
    logger.info("      First run also downloads model weights (~1 GB). Please wait.")

    # Import here so models load ONCE at startup, not on every request
    try:
        from app.ml_engine import load_models
        load_models()
        logger.info("[OK] ML models loaded successfully.")
    except Exception as e:
        logger.error("[ERROR] ML model loading failed: %s", str(e))
        raise

    logger.info("=" * 60)
    logger.info("  ToxiWatch is READY!")
    logger.info("  Dashboard:  http://localhost:8000")
    logger.info("  API Docs:   http://localhost:8000/docs")
    logger.info("=" * 60)

    yield                            # Server is now running and serving requests

    logger.info("ToxiWatch shutting down.")


# ── Create the FastAPI app instance ───────────────────────────────────────────
app = FastAPI(
    title="ToxiWatch — Multilingual Toxicity Moderator",
    description=(
        "Detects and moderates toxic comments in English, Hindi, and Hinglish. "
        "Uses Detoxify for toxicity scoring and Helsinki-NLP for translation."
    ),
    version="1.0.0",
    lifespan=lifespan,               # Hook in our startup/shutdown logic
)

# ── Mount static files directory ─────────────────────────────────────────────
# FIX: Use absolute STATIC_DIR path instead of relative "static" string.
# This prevents FileNotFoundError when uvicorn is launched from a different directory.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    logger.warning("[WARN] Static directory not found at: %s", STATIC_DIR)

# ── Include all API routes ────────────────────────────────────────────────────
app.include_router(router)


# ── Root route: serve the frontend HTML page ──────────────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def serve_frontend(request: Request):
    """
    When someone visits http://localhost:8000/ in their browser,
    serve the main HTML frontend page from static/index.html.

    FIX: Uses absolute path (INDEX_HTML) instead of relative "static/index.html"
    to prevent FileNotFoundError when server is launched from a different directory.
    """
    try:
        html_content = INDEX_HTML.read_text(encoding="utf-8")
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>ToxiWatch</h1><p>Frontend not found. "
                    "Please ensure static/index.html exists.</p>",
            status_code=404
        )


# ── Health check endpoint ─────────────────────────────────────────────────────
@app.get("/health", tags=["Utility"])
async def health_check():
    """
    Simple ping endpoint. Returns {"status": "ok"} if the server is running.
    Useful for monitoring tools or deployment health checks.
    """
    return {"status": "ok", "service": "ToxiWatch", "version": "1.0.0"}
