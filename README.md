# 🧠 The RAG Application — AI-Powered Quiz Generator

Upload a PDF → chunk & embed it → store vectors in a local FAISS index → retrieve relevant context →
Groq LLM writes **exactly 5 multiple-choice questions** grounded only in your document → answer → get scored.

## Architecture

```
PDF ──pypdf──► text ──chunker (1000 chars, 200 overlap)──► chunks
chunks ──SentenceTransformer (all-MiniLM-L6-v2)──► embeddings ──► FAISS (cosine / inner product)
FAISS ──semantic seed queries + random anchors──► diverse context ──► Groq LLM (JSON schema)
LLM JSON ──validate (5 Qs, 4 options, 1 answer, no dupes; retry ×3)──► quiz
answer key ──► st.session_state (server-side; never sent to the browser until submit)
```

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (upload, status, quiz, results, restart) |
| `rag_engine.py` | PDF extraction, chunking, embeddings, FAISS, retrieval, Groq generation + validation |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # then paste your GROQ_API_KEY
streamlit run app.py
```

Get a free key at <https://console.groq.com/keys>. The first run downloads the small embedding model (~90 MB).

## Configuration (`.env`)

| Variable | Default |
|---|---|
| `GROQ_API_KEY` | *(required)* |
| `GROQ_MODEL` | `openai/gpt-oss-120b` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |

> Note: Groq retired `llama-3.3-70b-versatile` on 2026‑08‑16. If your default model is ever retired,
> pick another from <https://console.groq.com/docs/models> and set `GROQ_MODEL`.

## Features covered

- PDF upload with status (Uploading → Processing → Ready), graceful errors for empty/invalid/encrypted/scanned PDFs
- Overlapping chunking, local FAISS index (also saved to `data/`)
- Strict-JSON quiz generation with validation + automatic retry, options shuffled to avoid letter bias
- Radio-button quiz in a form (no accidental reruns); answers persist in `st.session_state`
- Answer key stored server-side only; results show score `x/5 (y%)`, correct/incorrect, explanation, unanswered = wrong
- **Restart Quiz** generates a fresh quiz from the same document

## Push to GitHub

```bash
git init && git add . && git commit -m "RAG quiz app"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
`.env` is git-ignored, so your key stays private.
