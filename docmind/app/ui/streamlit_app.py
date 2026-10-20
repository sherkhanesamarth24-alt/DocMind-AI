import os
import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="DocMind", page_icon="📄", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 900px; padding-top: 2rem;}
.src {border-left: 3px solid #3b6fb6; padding: .4rem .8rem; margin: .4rem 0; background: rgba(59,111,182,.07); border-radius: 0 6px 6px 0; font-size: .9rem;}
.src b {color: #3b6fb6;}
</style>""", unsafe_allow_html=True)


def api(method, path, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=180, **kw)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(f"Request failed: {e.response.text}")
    except requests.ConnectionError:
        st.error(f"Can't reach the API at {API}. Start it with: uvicorn app.api.main:app --port 8000")
    return None


if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Your documents")
    files = st.file_uploader("Add PDF, DOCX or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if files and st.button("Index documents", type="primary", use_container_width=True):
        with st.spinner("Reading and indexing..."):
            res = api("POST", "/upload", files=[("files", (f.name, f.getvalue())) for f in files])
        for r in res or []:
            (st.error if "error" in r else st.success)(
                f"{r['filename']}: {r.get('error') or str(r['chunks']) + ' chunks indexed'}")

    docs = api("GET", "/documents") or []
    selected = []
    if not docs:
        st.info("No documents yet. Upload one to start asking questions.")
    for d in docs:
        c1, c2 = st.columns([5, 1])
        if c1.checkbox(f"{d['filename']} ({d['chunks']})", value=True, key=f"sel_{d['doc_id']}"):
            selected.append(d["doc_id"])
        if c2.button("✕", key=f"del_{d['doc_id']}", help="Delete this document"):
            api("DELETE", f"/documents/{d['doc_id']}")
            st.rerun()

    st.divider()
    st.subheader("Retrieval settings")
    hybrid = st.toggle("Hybrid search (BM25 + vectors)", value=True)
    rerank = st.toggle("Cross-encoder re-ranker", value=True)
    rewrite = st.toggle("Rewrite follow-up questions", value=True)
    top_k = st.slider("Passages sent to the model", 2, 10, 5)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- Main ----------
tab_chat, tab_eval = st.tabs(["Chat", "Evaluation"])


def show_sources(sources):
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"<div class='src'><b>[{s['id']}] {s['filename']}, page {s['page']}</b><br>{s['snippet']}…</div>",
                            unsafe_allow_html=True)


with tab_chat:
    st.title("DocMind")
    st.caption("Ask questions about your documents. Every answer links back to the page it came from.")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            show_sources(m.get("sources"))

    if q := st.chat_input("Ask something about your documents", disabled=not docs):
        with st.chat_message("user"):
            st.markdown(q)
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                res = api("POST", "/chat", json={
                    "question": q, "history": history, "doc_ids": selected or None,
                    "hybrid": hybrid, "rerank": rerank, "rewrite": rewrite, "top_k": top_k})
            if res:
                st.markdown(res["answer"])
                show_sources(res["sources"])
                if res.get("rewritten_query") and res["rewritten_query"] != q:
                    st.caption(f"Searched for: {res['rewritten_query']}")
        if res:
            st.session_state.messages += [{"role": "user", "content": q},
                                          {"role": "assistant", "content": res["answer"], "sources": res["sources"]}]

with tab_eval:
    st.subheader("Retrieval benchmark")
    results = api("GET", "/eval/results") or []
    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        st.bar_chart(pd.DataFrame(results).set_index("setup")[["hit@5", "mrr"]])
    else:
        st.info("No results yet. Run: python -m evaluation.eval_retrieval")
