import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()  # "gemini" or "claude"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))

CHROMA_DIR = os.getenv("CHROMA_DIR", str(BASE_DIR / "data" / "chroma"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))  # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # characters
TOP_K = int(os.getenv("TOP_K", "3"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_HISTORY_MESSAGES = 6
