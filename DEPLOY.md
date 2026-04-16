# 🚀 Deploying StudyRAG to Streamlit Cloud

## Prerequisites
- GitHub account
- Streamlit Community Cloud account (free): https://share.streamlit.io
- Groq API key: https://console.groq.com

---

## Step 1 — Push to GitHub

```bash
cd studyrag
git init
git add .
git commit -m "Initial StudyRAG commit"

# Create a new repo on GitHub (e.g. "studyrag"), then:
git remote add origin https://github.com/YOUR_USERNAME/studyrag.git
git branch -M main
git push -u origin main
```

> ⚠️ Make sure `.gitignore` is in place — it prevents `.env` and `vector_store/` from being pushed.

---

## Step 2 — Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io** and sign in with GitHub
2. Click **"New app"**
3. Fill in:
   - **Repository:** `YOUR_USERNAME/studyrag`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy!"**

---

## Step 3 — Add Your Groq API Key (Secrets)

In the Streamlit Cloud dashboard, before or after deploying:

1. Click **"⋮" (three dots)** → **"Settings"** next to your app
2. Go to **"Secrets"** tab
3. Paste this:

```toml
GROQ_API_KEY = "gsk_your_actual_key_here"
```

4. Click **"Save"** — the app restarts automatically

> The app reads this via `st.secrets["GROQ_API_KEY"]` — no `.env` file needed in the cloud.

---

## Step 4 — Handle the Vector Store on Cloud

Streamlit Cloud has **ephemeral storage** — files reset on each redeploy.

### Option A — Pre-embed & commit the vector store (simplest for ≤20 PDFs)

```bash
# Locally: index your 20 PDFs first
streamlit run app.py
# Click "Index Pre-loaded PDFs" in the sidebar
# This creates vector_store/ on disk

# Then remove vector_store from .gitignore temporarily:
# Comment out the "vector_store/" line in .gitignore

git add vector_store/
git commit -m "Add pre-built vector store"
git push

# Restore .gitignore afterwards
```

This is the **recommended approach** for a fixed set of 20 PDFs — the index is baked in and loads instantly.

### Option B — Re-index on each deploy (auto startup)

Add this to `app.py` in the startup block:

```python
# In the startup section, auto-index preloaded PDFs on first run
if not st.session_state.startup_done:
    if settings.GROQ_API_KEY:
        ingest_preloaded_pdfs()   # skips already-indexed files
    refresh_stats()
    st.session_state.startup_done = True
```

And make sure your PDFs are committed to `data/preloaded_pdfs/` in the repo.

---

## Step 5 — Commit Your PDFs (if using Option B)

```bash
# Add your 20 PDFs to the folder
cp /path/to/your/*.pdf data/preloaded_pdfs/

git add data/preloaded_pdfs/
git commit -m "Add 20 source PDFs"
git push
```

> PDFs under ~25 MB each are fine for GitHub. For larger files use Git LFS.

---

## Summary Checklist

- [ ] Code pushed to GitHub
- [ ] `GROQ_API_KEY` added in Streamlit Cloud Secrets
- [ ] PDFs in `data/preloaded_pdfs/` committed (or vector store pre-built)
- [ ] App deployed at `https://YOUR_USERNAME-studyrag.streamlit.app`

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError` | Check `requirements.txt` has all packages |
| `GROQ_API_KEY not set` | Add it in Streamlit Cloud Secrets tab |
| Vector store empty after deploy | Use Option A (commit vector store) |
| App crashes on large PDFs | Increase chunk size or split PDFs |
| Slow first response | HuggingFace model downloads on cold start — normal |
