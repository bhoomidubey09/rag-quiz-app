"""RAG pipeline: PDF -> chunks -> embeddings -> FAISS -> retrieval -> Groq quiz."""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from pypdf.errors import PyPdfError

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# llama-3.3-70b-versatile was retired by Groq on 2026-08-16; gpt-oss-120b is the
# recommended production replacement. Override with GROQ_MODEL in .env if needed.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

NUM_QUESTIONS = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_TEXT_CHARS = 300
MAX_CONTEXT_CHARS = 12000
INDEX_DIR = Path("data")


class PDFProcessingError(Exception):
    """Raised for user-facing PDF problems."""


class QuizGenerationError(Exception):
    """Raised when the LLM cannot produce a valid quiz."""


# --------------------------------------------------------------------------- PDF
def extract_text(file_obj) -> str:
    try:
        reader = PdfReader(file_obj)
        if reader.is_encrypted:
            try:
                if reader.decrypt("") == 0:
                    raise PDFProcessingError("The PDF is password-protected.")
            except PDFProcessingError:
                raise
            except Exception:
                raise PDFProcessingError("The PDF is encrypted and cannot be read.")
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
    except PDFProcessingError:
        raise
    except (PyPdfError, ValueError, OSError, KeyError, TypeError):
        raise PDFProcessingError("This file is not a valid or readable PDF.")
    except Exception as exc:  # noqa: BLE001
        raise PDFProcessingError(f"Could not read the PDF: {exc}")

    text = "\n\n".join(pages)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise PDFProcessingError(
            "No text could be extracted. The PDF may be empty or a scanned image (OCR is not supported)."
        )
    if len(text) < MIN_TEXT_CHARS:
        raise PDFProcessingError(
            f"The PDF contains too little text ({len(text)} characters) to generate a quiz."
        )
    return text


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Paragraph/sentence-aware splitter with character overlap."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
        else:  # split long paragraphs by sentence, hard-cut as a last resort
            for sent in re.split(r"(?<=[.!?])\s+", para):
                while len(sent) > size:
                    units.append(sent[:size])
                    sent = sent[size:]
                if sent:
                    units.append(sent)

    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 1 > size:
            chunks.append(current)
            tail = current[-overlap:]
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail  # start on a word boundary
            current = f"{tail} {unit}"
        else:
            current = f"{current} {unit}".strip()
    if current:
        chunks.append(current)
    return chunks


# ------------------------------------------------------------------ Vector store
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def embed(texts: list[str]) -> np.ndarray:
    vecs = get_embedder().encode(
        texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vecs, dtype="float32")


@dataclass
class VectorStore:
    chunks: list[str]
    index: faiss.Index = field(repr=False)

    @classmethod
    def build(cls, chunks: list[str], persist_name: str | None = None) -> "VectorStore":
        vectors = embed(chunks)
        index = faiss.IndexFlatIP(vectors.shape[1])  # cosine similarity (vectors are normalized)
        index.add(vectors)
        if persist_name:
            try:
                INDEX_DIR.mkdir(exist_ok=True)
                faiss.write_index(index, str(INDEX_DIR / f"{persist_name}.faiss"))
            except Exception:
                pass  # persistence is a convenience, never fatal
        return cls(chunks=chunks, index=index)

    def search(self, queries: list[str] | np.ndarray, k: int = 3) -> list[int]:
        qv = embed(queries) if isinstance(queries, list) else queries
        k = min(k, len(self.chunks))
        _, ids = self.index.search(qv, k)
        return [int(i) for row in ids for i in row if i >= 0]


SEED_QUERIES = [
    "main topic, purpose and overview of the document",
    "key definitions and important concepts",
    "important facts, figures, dates and names",
    "how processes, methods or systems work step by step",
    "causes, effects, comparisons and reasons",
    "conclusions, results, advantages and limitations",
]


def retrieve_context(store: VectorStore, max_chunks: int = 10) -> str:
    """Retrieve diverse relevant chunks (semantic seeds + random anchors for variety)."""
    picked: list[int] = []
    for qid in store.search(SEED_QUERIES, k=2):
        if qid not in picked:
            picked.append(qid)
    # random anchor chunks -> a different quiz on each "Generate"
    n = len(store.chunks)
    anchors = random.sample(range(n), k=min(3, n))
    anchor_vecs = np.asarray(
        [store.index.reconstruct(i) for i in anchors], dtype="float32"
    )
    for qid in store.search(anchor_vecs, k=2):
        if qid not in picked:
            picked.append(qid)
    random.shuffle(picked)
    picked = sorted(picked[:max_chunks])  # keep document order for coherence

    parts, total = [], 0
    for i in picked:
        piece = f"[Excerpt {i + 1}]\n{store.chunks[i]}"
        if total + len(piece) > MAX_CONTEXT_CHARS:
            break
        parts.append(piece)
        total += len(piece)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- LLM
SYSTEM_PROMPT = f"""You are an exam writer. Create exactly {NUM_QUESTIONS} multiple-choice questions \
using ONLY the document excerpts provided by the user.

Rules:
- Use only facts stated in the excerpts. Never use outside knowledge and never invent facts.
- Questions must be distinct, relevant to the document's content, and of easy-to-moderate difficulty.
- Cover different parts of the excerpts; avoid trivia about formatting, page numbers or authors.
- Each question has exactly four options labelled A, B, C, D with exactly ONE correct answer.
- Wrong options must be plausible but clearly incorrect according to the excerpts. No "All of the above"/"None of the above".
- Do not mention "excerpt", "passage" or "text" in the question wording.
- "explanation": one or two sentences explaining why the answer is correct, grounded in the excerpts.
- Respond with JSON only, in this shape:
{{"questions":[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"answer":"A","explanation":"..."}}]}}"""

JSON_SCHEMA = {
    "name": "quiz",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {
                            "type": "object",
                            "properties": {k: {"type": "string"} for k in "ABCD"},
                            "required": list("ABCD"),
                            "additionalProperties": False,
                        },
                        "answer": {"type": "string", "enum": list("ABCD")},
                        "explanation": {"type": "string"},
                    },
                    "required": ["question", "options", "answer", "explanation"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["questions"],
        "additionalProperties": False,
    },
}


def _client():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key.startswith("your_"):
        raise QuizGenerationError("GROQ_API_KEY is missing. Add it to your .env file and restart the app.")
    from groq import Groq

    return Groq(api_key=api_key)


def _call_llm(client, context: str, temperature: float) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Document excerpts:\n\n{context}\n\nWrite the quiz now."},
    ]
    base = dict(model=GROQ_MODEL, messages=messages, temperature=temperature, max_completion_tokens=4000)
    if GROQ_MODEL.startswith("openai/gpt-oss"):
        base["reasoning_effort"] = "low"

    # Prefer strict JSON-schema output, degrade gracefully if the model doesn't support it.
    formats = [
        {"type": "json_schema", "json_schema": JSON_SCHEMA},
        {"type": "json_object"},
        None,
    ]
    last_exc: Exception | None = None
    for fmt in formats:
        kwargs = dict(base)
        if fmt:
            kwargs["response_format"] = fmt
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            last_exc = exc
            if any(w in msg for w in ("response_format", "json_schema", "structured", "does not support")):
                continue  # try a weaker output mode
            raise
    raise last_exc  # type: ignore[misc]


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found")
        return json.loads(raw[start : end + 1])


def validate_quiz(data: dict) -> list[dict]:
    qs = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(qs, list) or len(qs) != NUM_QUESTIONS:
        raise ValueError(f"Expected exactly {NUM_QUESTIONS} questions")
    clean, seen = [], set()
    for q in qs:
        text = str(q.get("question", "")).strip()
        opts = q.get("options")
        ans = str(q.get("answer", "")).strip().upper()[:1]
        expl = str(q.get("explanation", "")).strip()
        if not text or not isinstance(opts, dict) or set(opts) != set("ABCD"):
            raise ValueError("Malformed question or options")
        opts = {k: str(v).strip() for k, v in opts.items()}
        if any(not v for v in opts.values()) or len({v.lower() for v in opts.values()}) != 4:
            raise ValueError("Options must be four distinct non-empty strings")
        if ans not in "ABCD" or not ans:
            raise ValueError("Answer must be one of A-D")
        key = re.sub(r"\W+", " ", text.lower()).strip()
        if key in seen:
            raise ValueError("Duplicate questions")
        seen.add(key)
        clean.append({"question": text, "options": opts, "answer": ans, "explanation": expl or "See the document."})
    return clean


def _shuffle_options(q: dict) -> dict:
    """LLMs favour certain letters; shuffle so the answer position is random."""
    items = list(q["options"].items())
    correct_text = q["options"][q["answer"]]
    random.shuffle(items)
    new_opts = {letter: text for letter, (_, text) in zip("ABCD", items)}
    new_ans = next(l for l, t in new_opts.items() if t == correct_text)
    return {**q, "options": new_opts, "answer": new_ans}


def generate_quiz(store: VectorStore, max_attempts: int = 3) -> tuple[list[dict], list[dict]]:
    """Returns (public_questions, answer_key). The key never leaves server-side state."""
    client = _client()
    last_error = "unknown error"
    for attempt in range(max_attempts):
        context = retrieve_context(store)
        try:
            raw = _call_llm(client, context, temperature=0.4 + 0.15 * attempt)
            quiz = [_shuffle_options(q) for q in validate_quiz(_parse_json(raw))]
        except (ValueError, KeyError, AttributeError, TypeError) as exc:
            last_error = str(exc)
            continue
        except Exception as exc:  # API / network error
            msg = str(exc)
            if "429" in msg or "rate" in msg.lower():
                raise QuizGenerationError("Groq rate limit reached. Please wait a moment and try again.")
            if "401" in msg or "invalid api key" in msg.lower():
                raise QuizGenerationError("Groq rejected the API key. Check GROQ_API_KEY in your .env file.")
            if "model" in msg.lower() and ("not found" in msg.lower() or "decommission" in msg.lower()):
                raise QuizGenerationError(f"Model '{GROQ_MODEL}' is unavailable. Set GROQ_MODEL in .env. ({msg})")
            last_error = msg
            continue
        public = [{"question": q["question"], "options": q["options"]} for q in quiz]
        key = [{"answer": q["answer"], "explanation": q["explanation"]} for q in quiz]
        return public, key
    raise QuizGenerationError(f"Could not generate a valid quiz after {max_attempts} attempts ({last_error}).")
