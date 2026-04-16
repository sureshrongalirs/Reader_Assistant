"""
app.py  —  StudyRAG main Streamlit application
Run:  streamlit run app.py
"""

import os
import sys
import tempfile
import logging

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from src.config.settings import settings
from src.ingestion.ingestion import (
    ingest_preloaded_pdfs,
    ingest_uploaded_files,
    get_collection_stats,
)
from src.agents.crew import run_analytical_query, detect_query_type

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Pull API key from Streamlit secrets if available (cloud deploy) ───────────
if not settings.GROQ_API_KEY:
    try:
        settings.GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
        os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StudyAssistant · PDF Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLING  —  clean light theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;800&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg:       #f8f9fc;
  --surface:  #ffffff;
  --surface2: #f1f3f8;
  --border:   #e2e5ed;
  --accent:   #6c47ff;
  --accent2:  #0ea5e9;
  --gold:     #d97706;
  --text:     #1a1d27;
  --muted:    #6b7280;
  --success:  #16a34a;
  --danger:   #dc2626;
  --warning:  #d97706;
  --radius:   12px;
  --shadow:   0 1px 4px rgba(0,0,0,.07), 0 4px 16px rgba(0,0,0,.04);
}
html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif !important;
}
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
  box-shadow: 2px 0 12px rgba(0,0,0,.04) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
h1 {
  font-family: 'Syne', sans-serif !important; font-size: 2rem !important;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
  letter-spacing: -.5px;
}
h2, h3 { font-family: 'Syne', sans-serif !important; color: var(--text) !important; }
[data-testid="stChatMessage"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow) !important;
  margin-bottom: 12px !important;
  padding: 16px 20px !important;
}
[data-testid="stChatInput"] textarea {
  background: var(--surface) !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif !important;
  border-color: var(--border) !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(108,71,255,.12) !important;
}
.stButton > button {
  background: linear-gradient(135deg, var(--accent), #4f35cc) !important;
  color: #fff !important; font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important; border: none !important;
  border-radius: 8px !important; padding: 8px 20px !important;
  box-shadow: 0 2px 8px rgba(108,71,255,.25) !important;
  transition: opacity .18s, transform .12s !important;
}
.stButton > button:hover { opacity:.92 !important; transform: translateY(-1px) !important; }
[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: var(--radius) !important;
}
[data-testid="stSelectbox"] > div > div {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  border-radius: 8px !important; color: var(--text) !important;
}
.streamlit-expanderHeader {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important; font-size: .82rem !important;
  color: var(--muted) !important;
}
.streamlit-expanderContent {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important; border-radius: 0 0 8px 8px !important;
}
.badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 10px; border-radius: 20px; font-size: .72rem; font-weight: 600;
}
.b-purple { background: rgba(108,71,255,.1);  color: #6c47ff; border: 1px solid rgba(108,71,255,.25); }
.b-blue   { background: rgba(14,165,233,.1);  color: #0284c7; border: 1px solid rgba(14,165,233,.25); }
.b-gold   { background: rgba(217,119,6,.1);   color: #d97706; border: 1px solid rgba(217,119,6,.25);  }
.b-green  { background: rgba(22,163,74,.1);   color: #16a34a; border: 1px solid rgba(22,163,74,.25);  }
.b-red    { background: rgba(220,38,38,.1);   color: #dc2626; border: 1px solid rgba(220,38,38,.25);  }
.src-pill {
  display: inline-block;
  background: rgba(14,165,233,.08); color: #0284c7;
  border: 1px solid rgba(14,165,233,.2); border-radius: 20px;
  padding: 2px 10px; font-size: .72rem; margin: 2px 2px 2px 0; font-weight: 500;
}
.status-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); box-shadow: var(--shadow);
  padding: 16px 22px; display: flex; align-items: center; gap: 14px;
}
.status-num { font-family: 'Syne', sans-serif; font-size: 1.8rem; color: var(--accent); line-height:1; }
.status-lbl { font-size: .7rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-top: 3px; }
.qtype-tag {
  display: inline-block; padding: 2px 10px; border-radius: 6px;
  font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em;
}
.qt-qa        { background: rgba(108,71,255,.1);  color: #6c47ff; }
.qt-summarise { background: rgba(217,119,6,.1);   color: #d97706; }
.qt-compare   { background: rgba(14,165,233,.1);  color: #0284c7; }
.qt-extract   { background: rgba(22,163,74,.1);   color: #16a34a; }
.conf-high   { color: var(--success); font-weight: 600; }
.conf-medium { color: var(--warning); font-weight: 600; }
.conf-low    { color: var(--danger);  font-weight: 600; }
hr { border-color: var(--border) !important; margin: 18px 0 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "chat_history": [],
    "index_ready":  False,
    "stats":        {"total_vectors": 0, "file_names": [], "total_files": 0},
    "startup_done": False,
    "tmp_dir":      None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
QTYPE_LABELS = {
    "qa":        ("Q&A",       "qt-qa"),
    "summarise": ("Summarise", "qt-summarise"),
    "compare":   ("Compare",   "qt-compare"),
    "extract":   ("Extract",   "qt-extract"),
}
QTYPE_HINTS = {
    "qa":        "💡 e.g. *What does [document] say about X?*",
    "summarise": "💡 e.g. *Summarise the key findings across all reports*",
    "compare":   "💡 e.g. *Compare results between [doc A] and [doc B]*",
    "extract":   "💡 e.g. *List all revenue figures mentioned in the PDFs*",
}

def qtype_badge(qt):
    label, cls = QTYPE_LABELS.get(qt, ("Q&A", "qt-qa"))
    return f'<span class="qtype-tag {cls}">{label}</span>'

def conf_span(conf):
    c = (conf or "medium").lower()
    dots = {"high": "●●●", "medium": "●●○", "low": "●○○"}.get(c, "●●○")
    return f'<span class="conf-{c}">{dots} {c.title()}</span>'

def source_pills(sources):
    return "".join(f'<span class="src-pill">📄 {s}</span>' for s in sources if s)

def refresh_stats():
    st.session_state.stats = get_collection_stats()
    st.session_state.index_ready = st.session_state.stats["total_vectors"] > 0

def ensure_tmp_dir():
    if not st.session_state.tmp_dir or not os.path.isdir(st.session_state.tmp_dir):
        st.session_state.tmp_dir = tempfile.mkdtemp(prefix="studyrag_uploads_")
    return st.session_state.tmp_dir

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.startup_done:
    refresh_stats()
    st.session_state.startup_done = True

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  —  PDF management only; no API key / model / tech details shown
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Pre-loaded PDFs")
    st.caption(f"Folder: `{settings.PRELOADED_PDFS_DIR}/`")
    st.caption("Place up to 20 PDFs in this folder, then click below.")

    if st.button("🔄 Index Pre-loaded PDFs", use_container_width=True):
        if not settings.GROQ_API_KEY:
            st.error("❌ GROQ_API_KEY not configured.")
        else:
            with st.spinner("Scanning & indexing…"):
                try:
                    result = ingest_preloaded_pdfs()
                    refresh_stats()
                    n_new  = len(result["indexed"])
                    n_skip = len(result["skipped"])
                    if n_new:
                        st.success(f"✅ {n_new} new file(s) indexed, {n_skip} skipped")
                    else:
                        st.info(f"ℹ️ All {n_skip} file(s) already indexed")
                except Exception as e:
                    st.error(f"❌ {e}")
                    logger.exception("Pre-load error")

    st.markdown("---")
    st.markdown("## ⬆️ Upload New PDFs")
    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("⚡ Add to Index", use_container_width=True):
        if not settings.GROQ_API_KEY:
            st.error("❌ GROQ_API_KEY not configured.")
        elif not uploaded:
            st.warning("⚠️ Select at least one PDF first.")
        else:
            prog = st.progress(0, text="Saving files…")
            try:
                prog.progress(30, text="Embedding & indexing…")
                result = ingest_uploaded_files(uploaded, ensure_tmp_dir())
                refresh_stats()
                prog.progress(100, text="Done!")
                n_new  = len(result["indexed"])
                n_skip = len(result["skipped"])
                msg = f"✅ **{n_new}** file(s) added"
                if n_skip:
                    msg += f", {n_skip} already existed"
                st.success(msg)
            except Exception as e:
                prog.empty()
                st.error(f"❌ {e}")
                logger.exception("Upload error")

    st.markdown("---")

    stats = st.session_state.stats
    if stats["file_names"]:
        st.markdown(f"**📑 {stats['total_files']} indexed PDF(s)**")
        for fname in stats["file_names"]:
            st.markdown(
                f'<span class="badge b-blue">PDF</span> '
                f'<span style="font-size:.8rem;color:#374151">{fname}</span>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<span style="color:#9ca3af;font-size:.82rem">No files indexed yet</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown(
        '<div style="color:#9ca3af;font-size:.7rem;line-height:1.9;margin-top:6px">'
        '① Add PDFs → folder or upload above<br>'
        '② Click the Index button<br>'
        '③ Ask questions in the chat<br><br>'
        '🔒 Index persists between restarts<br>'
        '♻️ Existing files are never re-embedded'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 📚 StudyRAG")
st.caption("Ask questions about your PDF library — Q&A · Summarise · Compare · Extract")

# ── Status row: PDFs Indexed + Ready status only ──────────────────────────────
stats = st.session_state.stats
rdy   = st.session_state.index_ready
s1, s2, _spacer = st.columns([1, 1, 2])

with s1:
    st.markdown(
        f'<div class="status-card">'
        f'<div><div class="status-num">{stats["total_files"]}</div>'
        f'<div class="status-lbl">PDFs Indexed</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with s2:
    bcls = "b-green" if rdy else "b-red"
    blbl = "✓ Ready" if rdy else "✗ No Index"
    st.markdown(
        f'<div class="status-card">'
        f'<span class="badge {bcls}" style="font-size:.85rem;padding:6px 16px">{blbl}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Query mode selector ───────────────────────────────────────────────────────
col_mode, col_hint = st.columns([1, 3])
with col_mode:
    query_mode = st.selectbox(
        "Query mode",
        ["Auto-detect", "Q&A", "Summarise", "Compare", "Extract Data"],
        help="Auto-detect picks the right mode from your question wording",
    )
with col_hint:
    mode_map      = {"Q&A": "qa", "Summarise": "summarise",
                     "Compare": "compare", "Extract Data": "extract"}
    selected_mode = mode_map.get(query_mode)
    hint = QTYPE_HINTS.get(selected_mode,
           "💡 Auto-detects Q&A, Summarise, Compare, or Extract from your question")
    st.markdown(
        f'<div style="padding:12px 0 0;color:#6b7280;font-size:.85rem">{hint}</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            qt   = msg.get("query_type", "qa")
            conf = msg.get("confidence", "medium")
            srcs = [s for s in msg.get("sources", []) if s]
            st.markdown(
                f'<div style="margin-top:10px;display:flex;flex-wrap:wrap;align-items:center;gap:8px">'
                f'{qtype_badge(qt)} '
                f'<span style="font-size:.72rem;color:#9ca3af">Confidence: {conf_span(conf)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if srcs:
                st.markdown(
                    f'<div style="margin-top:8px">'
                    f'<span style="font-size:.75rem;color:#9ca3af">Sources: </span>'
                    f'{source_pills(srcs)}</div>',
                    unsafe_allow_html=True,
                )
            rat  = msg.get("rationale", "")
            tool = msg.get("tool_used", "")
            if rat or tool:
                with st.expander("🔍 Reasoning & tool"):
                    if tool: st.markdown(f"**Tool:** `{tool}`")
                    if rat:  st.markdown(f"**Rationale:** {rat}")

# ── Chat input ────────────────────────────────────────────────────────────────
user_prompt = st.chat_input("Ask anything about your documents…")

if user_prompt:
    if not settings.GROQ_API_KEY:
        st.warning("⚠️ GROQ_API_KEY is not configured.")
    elif not st.session_state.index_ready:
        st.warning("⚠️ No documents indexed yet. Use the sidebar to add PDFs.")
    else:
        with st.chat_message("user"):
            st.markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        with st.chat_message("assistant"):
            detected = selected_mode or detect_query_type(user_prompt)
            label, _ = QTYPE_LABELS.get(detected, ("Q&A", "qt-qa"))
            with st.spinner(f"Running {label} analysis…"):
                try:
                    result     = run_analytical_query(
                        query=user_prompt,
                        chat_history=st.session_state.chat_history,
                        query_type=selected_mode,
                    )
                    answer     = result.get("answer", "(No response)")
                    qt         = result.get("query_type", "qa")
                    tool       = result.get("tool_used", "RAG + CrewAI")
                    rationale  = result.get("rationale", "")
                    sources    = [s for s in result.get("sources", []) if s]
                    confidence = result.get("confidence", "medium")
                except Exception as e:
                    answer     = f"❌ Error: {e}"
                    qt         = "qa"
                    tool       = "Error"
                    rationale  = str(e)
                    sources    = []
                    confidence = "low"
                    logger.exception("Query error")

            st.markdown(answer)
            st.markdown(
                f'<div style="margin-top:10px;display:flex;flex-wrap:wrap;align-items:center;gap:8px">'
                f'{qtype_badge(qt)} '
                f'<span style="font-size:.72rem;color:#9ca3af">Confidence: {conf_span(confidence)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if sources:
                st.markdown(
                    f'<div style="margin-top:8px">'
                    f'<span style="font-size:.75rem;color:#9ca3af">Sources: </span>'
                    f'{source_pills(sources)}</div>',
                    unsafe_allow_html=True,
                )
            if tool or rationale:
                with st.expander("🔍 Reasoning & tool"):
                    if tool: st.markdown(f"**Tool:** `{tool}`")
                    if rationale: st.markdown(f"**Rationale:** {rationale}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "query_type": qt,
            "tool_used": tool,
            "rationale": rationale,
            "sources": sources,
            "confidence": confidence,
        })
