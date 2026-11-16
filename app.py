"""Streamlit interface for the Zyro Dynamics HR policy assistant."""

from __future__ import annotations

import streamlit as st

from rag_pipeline import build_pipeline, ask_question

st.set_page_config(
    page_title="Zyro People | HR Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#15243a; --muted:#62718a; --blue:#2457d6; --line:#e5eaf2; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#f7f9fd 0%,#f5f7fb 100%); color:var(--ink); }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
.brand { font-family:Manrope,sans-serif; font-size:1.1rem; font-weight:800; color:var(--ink); letter-spacing:-.03em; }
.brand-mark { color:var(--blue); font-size:1.35rem; margin-right:.45rem; }
.hero { padding:2.2rem 0 1rem; }
.eyebrow { color:var(--blue); text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:700; }
.hero h1 { font-family:Manrope,sans-serif; font-size:2.45rem; line-height:1.15; letter-spacing:-.045em; color:var(--ink); margin:.45rem 0; }
.hero p { color:var(--muted); font-size:1.03rem; max-width:660px; }
.trust { border:1px solid var(--line); background:#fff; border-radius:14px; padding:1rem 1.1rem; color:var(--muted); font-size:.88rem; }
.stChatMessage { background:#fff; border:1px solid var(--line); border-radius:16px; padding:.4rem .8rem; }
.stChatInput textarea { border-radius:14px !important; }
div[data-testid="stMetric"] { background:#f7f9fd; border:1px solid var(--line); padding:.8rem; border-radius:12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    return build_pipeline()


with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">✦</span>Zyro People</div>', unsafe_allow_html=True)
    st.caption("INTERNAL KNOWLEDGE ASSISTANT")
    st.divider()
    st.markdown("#### Your HR knowledge base")
    st.markdown("Answers are grounded in the company documents available to this assistant.")
    st.markdown("\n**Try asking**")
    examples = [
        "How many casual leave days are available?",
        "What is the work from home policy?",
        "How do I report a security concern?",
    ]
    for example in examples:
        if st.button(example, use_container_width=True, key=f"example_{example}"):
            st.session_state.pending_question = example
    st.divider()
    st.caption("For complex or personal cases, contact your HR representative.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">People &amp; policy support</div>
  <h1>Your workplace answers,<br>all in one place.</h1>
  <p>Ask a question about Zyro Dynamics policies and benefits. The assistant searches the HR documents and shows where its answer came from.</p>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="trust">🔒 &nbsp; Answers are based on the connected HR document library. If a policy detail is not available, the assistant will say so.</div>', unsafe_allow_html=True)

try:
    with st.spinner("Preparing the HR knowledge base…"):
        pipeline = load_pipeline()
except Exception as exc:
    st.error("The assistant could not start. Check the setup details below.")
    with st.expander("Setup details", expanded=True):
        st.code(str(exc))
        st.markdown("Add `GROQ_API_KEY` to `.env` (or deployment secrets), install `requirements.txt`, and make sure the PDF folder is present.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    page = f" · page {source['page']}" if source.get("page") else ""
                    st.caption(f"📄 {source['file']}{page}")

question = st.chat_input("Ask about leave, benefits, workplace policies…")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the HR documents…"):
            try:
                result = ask_question(question, pipeline)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources"):
                        for source in result["sources"]:
                            page = f" · page {source['page']}" if source.get("page") else ""
                            st.caption(f"📄 {source['file']}{page}")
                st.session_state.messages.append({
                    "role": "assistant", "content": result["answer"], "sources": result["sources"]
                })
            except Exception as exc:
                st.error(f"I couldn't complete that request: {exc}")

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
with st.expander("About this assistant"):
    st.write("This assistant retrieves passages from the configured PDF collection and uses them to draft a response. Check the cited policy before acting on an important or personal matter.")
