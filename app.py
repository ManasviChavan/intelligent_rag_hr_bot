"""Streamlit interface for the Zyro Dynamics HR policy assistant."""

from __future__ import annotations

import streamlit as st

from rag_pipeline import ask_question, build_pipeline

st.set_page_config(
    page_title="Zyro People | HR Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root, [data-theme="dark"], [data-testid="stAppViewContainer"] {
  --ink:#15243a; --muted:#53627a; --blue:#2457d6; --line:#dce3ee;
  --text-color:#15243a; --background-color:#f5f7fb; --secondary-background-color:#ffffff;
  --primary-color:#2457d6; --input-focus-color:#2457d6;
}
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color-scheme:light; }
.stApp, [data-testid="stAppViewContainer"] { background:linear-gradient(180deg,#f7f9fd 0%,#f3f6fb 100%); color:#15243a; }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, label { color:#15243a; }
.brand { font-family:Manrope,sans-serif; font-size:1.1rem; font-weight:800; color:#15243a; letter-spacing:-.03em; }
.brand-mark { display:inline-flex; align-items:center; justify-content:center; width:1.6rem; height:1.6rem; margin-right:.45rem; border-radius:8px; background:#eaf0ff; color:#2457d6; font-size:.66rem; letter-spacing:0; }
.hero { padding:2.2rem 0 1rem; }
.eyebrow { color:#2457d6; text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:700; }
.hero h1 { font-family:Manrope,sans-serif; font-size:2.45rem; line-height:1.15; letter-spacing:-.045em; color:#15243a; margin:.45rem 0; }
.hero p { color:#53627a; font-size:1.03rem; max-width:660px; }
.trust { border:1px solid var(--line); background:#fff; border-radius:14px; padding:1rem 1.1rem; color:#43516a; font-size:.92rem; }
div[data-testid="stChatMessage"] { background:#fff; border:1px solid #dce3ee; border-radius:16px; padding:.65rem .9rem; box-shadow:0 2px 8px rgba(20,40,75,.035); }
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li { color:#15243a !important; font-size:1rem; line-height:1.65; }
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong { color:#101d31 !important; }
.stChatInput textarea { border-radius:14px !important; color:#15243a !important; background:#fff !important; caret-color:#2457d6; }
.stChatInput textarea::placeholder { color:#66758b !important; opacity:1; }
button[kind="secondary"] { color:#15243a !important; }
div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary * { color:#15243a !important; }
div[data-testid="stCaptionContainer"] p { color:#53627a !important; }
div[data-testid="stMetric"] { background:#f7f9fd; border:1px solid var(--line); padding:.8rem; border-radius:12px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    return build_pipeline()


with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">ZD</span>Zyro People</div>', unsafe_allow_html=True)
    st.caption("INTERNAL KNOWLEDGE ASSISTANT")
    st.divider()
    st.markdown("#### Your HR knowledge base")
    st.markdown("Answers are verified against the company HR PDF library.")
    st.markdown("**Try asking**")
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

st.markdown(
    """
<div class="hero">
  <div class="eyebrow">People &amp; policy support</div>
  <h1>Your workplace answers,<br>all in one place.</h1>
  <p>Ask about Zyro Dynamics policies and benefits. Each answer is grounded in the HR PDFs and shows its source.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="trust">Answers are checked against the connected HR PDF library. If a detail is not supported by those documents, the assistant will say so.</div>',
    unsafe_allow_html=True,
)

try:
    with st.spinner("Preparing the HR knowledge base..."):
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
            with st.expander("Verified PDF sources"):
                for source in message["sources"]:
                    page = f" - page {source['page']}" if source.get("page") else ""
                    st.caption(f"{source['file']}{page}")

question = st.chat_input("Ask about leave, benefits, workplace policies...")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the HR PDFs..."):
            try:
                result = ask_question(question, pipeline)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Verified PDF sources"):
                        for source in result["sources"]:
                            page = f" - page {source['page']}" if source.get("page") else ""
                            st.caption(f"{source['file']}{page}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except Exception as exc:
                st.error(f"I couldn't complete that request: {exc}")

with st.expander("About this assistant"):
    st.write("The assistant searches only the configured HR PDF collection. It verifies source quotations before showing an answer. Contact HR for help with cases that require personal context or an official decision.")
