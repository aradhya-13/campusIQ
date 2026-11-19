import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import rag, store
from .config import ALLOWED_EXTENSIONS, BASE_DIR, MAX_UPLOAD_MB, TOP_K
from .ingest import ingest_file

app = FastAPI(title="Document Chat (RAG)")

STATIC_DIR = BASE_DIR / "static"


# ── Models ──────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[Message] = []


class SummariseRequest(BaseModel):
    history: list[Message] = []




class IngestURLRequest(BaseModel):
    url: str = Field(min_length=4, max_length=2000)


# ── Static ───────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# ── Documents ────────────────────────────────────────────────────────────────

@app.get("/api/documents")
def documents():
    return store.list_documents()


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        name = Path(f.filename or "").name
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            results.append({"name": name, "error": f"Unsupported type {ext or '(none)'}"})
            continue
        data = await f.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            results.append({"name": name, "error": f"Larger than {MAX_UPLOAD_MB} MB"})
            continue
        try:
            chunks = ingest_file(name, data)
            results.append({"name": name, "chunks": chunks})
        except Exception as e:
            results.append({"name": name, "error": str(e)})
    return results


@app.delete("/api/documents/{name}")
def delete_document(name: str):
    store.delete_document(Path(name).name)
    return {"deleted": name}


# ── URL ingestion (web pages + YouTube) ─────────────────────────────────────

def _ingest_youtube(video_id: str, url: str) -> dict:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except (NoTranscriptFound, TranscriptsDisabled) as e:
        raise ValueError(f"No transcript available for this video: {e}")
    text = " ".join(seg["text"] for seg in transcript)
    name = f"youtube_{video_id}.txt"
    chunks = ingest_file(name, text.encode("utf-8"))
    return {"name": name, "chunks": chunks, "type": "youtube", "url": url}


def _ingest_webpage(url: str) -> dict:
    import urllib.request
    from bs4 import BeautifulSoup
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read()
    soup = BeautifulSoup(html, "html.parser")
    # Remove nav, footer, script, style noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    # Collapse blank lines
    lines = [l for l in text.splitlines() if l.strip()]
    text = "\n".join(lines)
    if len(text) < 100:
        raise ValueError("Page has too little readable text.")
    # Use domain + path as name
    from urllib.parse import urlparse
    parsed = urlparse(url)
    slug = (parsed.netloc + parsed.path).strip("/").replace("/", "_")[:60]
    name = f"web_{slug}.txt"
    chunks = ingest_file(name, text.encode("utf-8"))
    return {"name": name, "chunks": chunks, "type": "webpage", "url": url}


@app.post("/api/ingest-url")
def ingest_url(req: IngestURLRequest):
    url = req.url.strip()
    # Detect YouTube
    import re
    yt_patterns = [
        r"(?:youtube\.com/watch\?.*v=|youtu\.be/)([A-Za-z0-9_-]{11})",
    ]
    video_id = None
    for pat in yt_patterns:
        m = re.search(pat, url)
        if m:
            video_id = m.group(1)
            break
    try:
        if video_id:
            result = _ingest_youtube(video_id, url)
        else:
            result = _ingest_webpage(url)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch URL: {e}")
    return result


# ── API key ──────────────────────────────────────────────────────────────────

class SetKeyRequest(BaseModel):
    key: str
    admin_password: str = ""


@app.post("/api/set-key")
def set_key(req: SetKeyRequest):
    import app.config as cfg
    import app.rag as rag_module
    # On Railway (or any deployment where ADMIN_PASSWORD is set), require it.
    expected = os.environ.get("ADMIN_PASSWORD", "")
    if expected and req.admin_password != expected:
        raise HTTPException(status_code=403, detail="Incorrect admin password.")
    cfg.GEMINI_API_KEY = req.key.strip()
    rag_module.GEMINI_API_KEY = req.key.strip()
    rag_module._gemini_client = None
    return {"ok": True}


# ── Summarise ────────────────────────────────────────────────────────────────

@app.post("/api/summarise")
def summarise(req: SummariseRequest):
    if not req.history:
        raise HTTPException(status_code=400, detail="No history to summarise.")
    convo = "\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
        for m in req.history
    )

    def generate():
        try:
            msgs = [{"role": "user", "content":
                f"Summarise the following conversation in 3-5 bullet points, plain text, no markdown:\n\n{convo}"}]
            for token in rag._stream_gemini(msgs):
                yield json.dumps({"type": "token", "text": token}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ── Quiz ─────────────────────────────────────────────────────────────────────

class QuizRequest(BaseModel):
    topic: str = ""          # optional focus topic
    num_questions: int = 5

@app.post("/api/quiz")
def generate_quiz(req: QuizRequest):
    """Ask the LLM to build a quiz from the stored documents."""
    hits = store.query(req.topic or "main topics key concepts", 6)
    if not hits:
        raise HTTPException(status_code=422, detail="No documents uploaded yet.")

    context = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
    n = max(3, min(req.num_questions, 15))
    prompt = f"""You are a quiz maker. Using ONLY the content below, create {n} multiple-choice questions.

Return ONLY valid JSON in this exact format, nothing else:
{{
  "title": "Quiz title based on content",
  "questions": [
    {{
      "q": "Question text?",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "answer": "A",
      "explanation": "Brief explanation."
    }}
  ]
}}

Content:
{context}
"""
    full = ""
    for token in rag._stream_gemini([{"role": "user", "content": prompt}]):
        full += token

    # Extract JSON from response
    import re
    match = re.search(r'\{[\s\S]*\}', full)
    if not match:
        raise HTTPException(status_code=500, detail="Could not parse quiz from AI response.")
    try:
        quiz = json.loads(match.group())
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON in AI response.")
    return quiz


# ── PPT ──────────────────────────────────────────────────────────────────────

class PPTRequest(BaseModel):
    topic: str = ""
    num_slides: int = 6

@app.post("/api/generate-ppt")
def generate_ppt(req: PPTRequest):
    """Generate a .pptx from the stored documents."""
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    import io, re

    hits = store.query(req.topic or "main topics overview", 8)
    if not hits:
        raise HTTPException(status_code=422, detail="No documents uploaded yet.")

    context = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
    n = max(3, min(req.num_slides, 12))
    prompt = f"""You are a presentation designer. Using ONLY the content below, create {n} slides.

Return ONLY valid JSON, nothing else:
{{
  "title": "Presentation title",
  "slides": [
    {{
      "heading": "Slide heading",
      "bullets": ["Point one", "Point two", "Point three"]
    }}
  ]
}}

Content:
{context}
"""
    full = ""
    for token in rag._stream_gemini([{"role": "user", "content": prompt}]):
        full += token

    match = re.search(r'\{[\s\S]*\}', full)
    if not match:
        raise HTTPException(status_code=500, detail="Could not parse slides from AI response.")
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON in AI response.")

    # Build PPTX
    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)

    DARK  = RGBColor(0x18, 0x18, 0x1b)
    LIGHT = RGBColor(0xFF, 0xFF, 0xFF)
    ACC   = RGBColor(0xC5, 0xCF, 0xF9)   # periwinkle accent

    def _blank(prs):
        return prs.slides.add_slide(prs.slide_layouts[6])  # completely blank

    def _textbox(slide, text, l, t, w, h, size, bold=False, color=DARK, align="left"):
        from pptx.util import Emu
        from pptx.enum.text import PP_ALIGN
        txb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
        tf  = txb.text_frame
        tf.word_wrap = True
        p   = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if align == "left" else PP_ALIGN.CENTER
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        return txb

    def _rect(slide, l, t, w, h, fill):
        from pptx.util import Inches as I
        shp = slide.shapes.add_shape(1, I(l), I(t), I(w), I(h))  # 1 = rectangle
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
        shp.line.fill.background()
        return shp

    # ── Title slide ──
    sld = _blank(prs)
    _rect(sld, 0, 0, 13.33, 7.5, DARK)
    _rect(sld, 0, 5.8, 13.33, 1.7, ACC)
    _textbox(sld, data.get("title","Presentation"), 0.8, 2.2, 11.5, 2.0, 40, bold=True, color=LIGHT, align="left")
    _textbox(sld, "Generated by Document Chat", 0.8, 4.4, 10, 0.6, 14, color=RGBColor(0xA1,0xA1,0xAA), align="left")

    # ── Content slides ──
    for slide_data in data.get("slides", []):
        sld = _blank(prs)
        # subtle top bar
        _rect(sld, 0, 0, 13.33, 0.08, ACC)
        # heading
        _textbox(sld, slide_data.get("heading",""), 0.6, 0.35, 12, 0.9, 24, bold=True, color=DARK)
        # divider line
        _rect(sld, 0.6, 1.3, 11.5, 0.04, ACC)
        # bullets
        bullets = slide_data.get("bullets", [])
        y = 1.55
        for b in bullets[:6]:
            _textbox(sld, f"• {b}", 0.75, y, 11.5, 0.65, 16, color=RGBColor(0x52,0x52,0x5b))
            y += 0.75

    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)

    safe_title = re.sub(r'[^\w\s-]', '', data.get("title","presentation"))[:40].strip().replace(" ","_")
    filename = f"{safe_title or 'presentation'}.pptx"

    from fastapi.responses import Response
    return Response(
        content=buf.read(),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# ── Chat ─────────────────────────────────────────────────────────────────────

def _line(obj: dict) -> str:
    return json.dumps(obj) + "\n"


@app.post("/api/chat")
def chat(req: ChatRequest):
    hits = store.query(req.question, TOP_K)

    def generate():
        yield _line({
            "type": "sources",
            "sources": [
                {"id": i, "source": h["source"], "page": h["page"],
                 "score": h["score"], "text": h["text"][:400]}
                for i, h in enumerate(hits, start=1)
            ],
        })
        try:
            history = [m.model_dump() for m in req.history]
            for token in rag.stream_answer(req.question, hits, history):
                yield _line({"type": "token", "text": token})
        except RuntimeError as e:
            yield _line({"type": "error", "message": str(e)})
        except Exception as e:
            msg = str(e)
            if len(msg) > 300:
                msg = msg[:300] + "..."
            yield _line({"type": "error", "message": f"LLM request failed: {msg}"})
        yield _line({"type": "done"})

    return StreamingResponse(generate(), media_type="application/x-ndjson")
