"""
app.py
Streamlit chat UI for the E-commerce Support Assistant.

Run with: streamlit run app.py
Requires GROQ_API_KEY set in a .env file (see README).
"""

import streamlit as st
from dotenv import load_dotenv

from agent import SupportAgent

load_dotenv()  # loads GROQ_API_KEY from .env

st.set_page_config(page_title="Support Assistant", page_icon="🛒", layout="centered")

st.title("🛒 E-Commerce Support Assistant")
st.caption("RAG (FAQ/Policies) + Tool-Calling (Product & Order lookup) — powered by Groq/Llama 3.3")

@st.cache_resource
def load_agent():
    return SupportAgent(data_dir="data")

try:
    agent = load_agent()
except KeyError:
    st.error("GROQ_API_KEY not found. Add it to your .env file, then restart.")
    st.stop()

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []

with st.sidebar:
    st.subheader("Try a sample query")
    samples = [
        "Where is my order?",
        "How do I return a product?",
        "My payment failed but money was deducted.",
        "Do you charge for shipping?",
        "How do I contact customer support?",
        "Does the 24-inch monitor have HDMI?",
        "What's the status of order ORD1001?",
        "I want to return order ORD1003, the item arrived damaged",
        "Can I return order ORD1001? I changed my mind",
        "This isn't helping, I want to talk to a real person",
    ]
    clicked_sample = None
    for s in samples:
        if st.button(s, use_container_width=True):
            clicked_sample = s

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.display_messages = []
        st.session_state.agent_history = []
        st.rerun()

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about your order, a product, or our policies...")
final_input = clicked_sample or user_input

if final_input:
    st.session_state.display_messages.append({"role": "user", "content": final_input})
    with st.chat_message("user"):
        st.markdown(final_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.ask(final_input, history=st.session_state.agent_history)
        st.markdown(result["answer"])

    st.session_state.agent_history = result["history"]
    st.session_state.display_messages.append({
        "role": "assistant",
        "content": result["answer"],
    })