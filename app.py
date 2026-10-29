"""AI-Powered Quiz Generator — Streamlit front-end for the RAG pipeline."""
import hashlib
import io
import os

import streamlit as st

import rag_engine as rag

st.set_page_config(page_title="RAG Quiz Generator", page_icon="🧠", layout="wide")

# ----------------------------------------------------------------------- styling
st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 980px;}
.hero {background: linear-gradient(135deg,#4F46E5 0%,#7C3AED 100%); color:#fff;
       padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;}
.hero h1 {margin:0; font-size: 2rem; color:#fff;}
.hero p {margin:.4rem 0 0; opacity:.92; font-size:1.02rem;}
.qcard {background:#F8F9FC; border:1px solid #E5E7EB; border-radius:12px;
        padding: 1rem 1.2rem .4rem; margin-bottom: .8rem;}
.qtitle {font-weight:600; font-size:1.05rem; margin-bottom:.4rem;}
.scorebox {text-align:center; padding:1.2rem; border-radius:14px; margin:.6rem 0 1.2rem;
           background:#EEF2FF; border:1px solid #C7D2FE;}
.scorebox .big {font-size:2.6rem; font-weight:800; color:#4338CA;}
div[data-testid="stMetric"] {background:#F8F9FC; border:1px solid #E5E7EB; border-radius:12px; padding:.6rem 1rem;}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ session state
DEFAULTS = {
    "doc_hash": None, "doc_name": None, "store": None, "n_chunks": 0, "n_chars": 0,
    "status": "Waiting for PDF", "error": None,
    "quiz": None, "answer_key": None, "quiz_id": 0, "submitted": False, "results": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
S = st.session_state


def reset_quiz():
    S.quiz, S.answer_key, S.submitted, S.results = None, None, False, None
    for key in [k for k in S if str(k).startswith("q_")]:
        del S[key]


def reset_all():
    reset_quiz()
    for k, v in DEFAULTS.items():
        if k != "quiz_id":
            S[k] = v


# ---------------------------------------------------------------------- header
st.markdown(
    """<div class="hero"><h1>🧠 AI-Powered Quiz Generator</h1>
<p>Upload a PDF, and a Retrieval-Augmented Generation pipeline (Sentence Transformers + FAISS + Groq)
creates a 5-question multiple-choice quiz grounded only in your document.</p></div>""",
    unsafe_allow_html=True,
)

if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY", "").startswith("your_"):
    st.error("`GROQ_API_KEY` not found. Copy `.env.example` to `.env`, add your key, and restart the app.")

# ------------------------------------------------------------------ PDF upload
uploaded = st.file_uploader("Upload a PDF document", type=["pdf"], help="Text-based PDFs only (no scanned images).")

if uploaded is None:
    if S.doc_hash is not None:
        reset_all()
else:
    data = uploaded.getvalue()
    doc_hash = hashlib.sha256(data).hexdigest()
    if doc_hash != S.doc_hash:  # new document -> process once
        reset_all()
        S.doc_hash, S.doc_name = doc_hash, uploaded.name
        with st.status("Processing document…", expanded=True) as status:
            try:
                S.status = "Uploading"
                st.write("📄 Extracting text from PDF…")
                text = rag.extract_text(io.BytesIO(data))
                S.status = "Processing"
                st.write("✂️ Splitting into overlapping chunks…")
                chunks = rag.chunk_text(text)
                st.write(f"🔢 Embedding {len(chunks)} chunks and building FAISS index…")
                S.store = rag.VectorStore.build(chunks, persist_name=doc_hash[:12])
                S.n_chunks, S.n_chars, S.status = len(chunks), len(text), "Ready"
                status.update(label="Document ready ✅", state="complete", expanded=False)
            except rag.PDFProcessingError as exc:
                S.error, S.status = str(exc), "Failed"
                status.update(label="Processing failed", state="error")
            except Exception as exc:  # noqa: BLE001
                S.error, S.status = f"Unexpected error while processing the PDF: {exc}", "Failed"
                status.update(label="Processing failed", state="error")

if S.error:
    st.error(S.error)

# ------------------------------------------------------------------- dashboard
c1, c2, c3, c4 = st.columns(4)
c1.metric("Status", S.status)
c2.metric("Chunks indexed", S.n_chunks)
c3.metric("Questions", len(S.quiz) if S.quiz else 0)
c4.metric("Score", f"{S.results['score']}/{rag.NUM_QUESTIONS}" if S.results else "—")

if S.status == "Ready" and S.doc_name:
    st.success(f"**{S.doc_name}** processed — {S.n_chars:,} characters → {S.n_chunks} chunks stored in FAISS.")

st.divider()

# ---------------------------------------------------------------- quiz: generate
if S.store is not None and S.quiz is None:
    st.subheader("Generate your quiz")
    if st.button("✨ Generate Quiz", type="primary", use_container_width=True):
        with st.spinner("Retrieving context and asking the LLM…"):
            try:
                S.quiz, S.answer_key = rag.generate_quiz(S.store)
                S.quiz_id += 1
                S.submitted, S.results = False, None
            except rag.QuizGenerationError as exc:
                st.error(str(exc))
        if S.quiz:
            st.rerun()

# ------------------------------------------------------------------ quiz: answer
LETTERS = list("ABCD")

if S.quiz and not S.submitted:
    st.subheader("Answer the questions")
    with st.form("quiz_form"):
        for i, q in enumerate(S.quiz):
            st.markdown(f'<div class="qcard"><div class="qtitle">Q{i + 1}. {q["question"]}</div></div>',
                        unsafe_allow_html=True)
            st.radio(
                f"Question {i + 1}", LETTERS, index=None, key=f"q_{S.quiz_id}_{i}",
                format_func=lambda l, q=q: f"{l}. {q['options'][l]}", label_visibility="collapsed",
            )
        submitted = st.form_submit_button("✅ Submit Quiz", type="primary", use_container_width=True)

    if submitted:
        picks = [S.get(f"q_{S.quiz_id}_{i}") for i in range(len(S.quiz))]
        details, score = [], 0
        for q, key, pick in zip(S.quiz, S.answer_key, picks):
            ok = pick is not None and pick == key["answer"]  # unanswered is never correct
            score += ok
            details.append({"pick": pick, "correct": ok, "answer": key["answer"], "explanation": key["explanation"]})
        S.results = {"score": score, "details": details}
        S.submitted = True
        st.rerun()

# ---------------------------------------------------------------- quiz: results
if S.quiz and S.submitted and S.results:
    res, total = S.results, len(S.quiz)
    pct = round(100 * res["score"] / total)
    st.markdown(
        f'<div class="scorebox"><div>Final score</div><div class="big">{res["score"]}/{total} ({pct}%)</div></div>',
        unsafe_allow_html=True,
    )
    for i, (q, d) in enumerate(zip(S.quiz, res["details"])):
        st.markdown(f"**Q{i + 1}. {q['question']}**")
        for l in LETTERS:
            text = f"{l}. {q['options'][l]}"
            if l == d["answer"]:
                st.success(f"{text}  ✔ correct answer" + ("  (your answer)" if l == d["pick"] else ""))
            elif l == d["pick"]:
                st.error(f"{text}  ✖ your answer")
            else:
                st.markdown(f"&nbsp;&nbsp;&nbsp;{text}")
        if d["pick"] is None:
            st.warning("Not answered — counted as incorrect.")
        st.info(f"💡 {d['explanation']}")
        st.write("")

    if st.button("🔄 Restart Quiz", use_container_width=True):
        reset_quiz()
        st.rerun()
elif S.store is None and not S.error:
    st.info("Upload a PDF to get started.")
