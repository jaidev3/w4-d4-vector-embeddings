import io
import json
from typing import List

import requests
import streamlit as st

API_URL = "http://localhost:8000"  # FastAPI backend address

st.set_page_config(page_title="HR Onboarding Assistant", page_icon="🧑‍💼")

st.title("🧑‍💼 HR Onboarding Knowledge Assistant")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tab_upload, tab_chat = st.tabs(["📄 Upload HR Documents", "💬 Chat"])

# ----------------------------
# Upload Tab
# ----------------------------
with tab_upload:
    st.header("Upload HR Documents")
    uploaded_files = st.file_uploader(
        "Choose HR files (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Upload & Ingest"):
        files: List[tuple[str, tuple[str, io.BytesIO, str]]] = []
        for up_file in uploaded_files:
            bytes_io = io.BytesIO(up_file.read())
            files.append(("files", (up_file.name, bytes_io, up_file.type or "application/octet-stream")))

        with st.spinner("Uploading and ingesting..."):
            try:
                r = requests.post(f"{API_URL}/upload", files=files, timeout=300)
                r.raise_for_status()
                resp = r.json()
                st.success("Upload successful ✅")
                st.write(resp)
            except Exception as e:
                st.error(f"Upload failed: {e}")

# ----------------------------
# Chat Tab
# ----------------------------
with tab_chat:
    st.header("Ask a question about company policies")

    if "messages" not in st.session_state:
        st.session_state.messages = []  # List of {"role": ..., "content": ...}

    # Display previous chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input (bottom)
    if prompt := st.chat_input("Type your question ..."):
        # Add user message to history & display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/chat",
                        json={"query": prompt},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    answer = data.get("response", "(no answer)")
                    citations = data.get("citations", [])

                    # Format answer with citations
                    formatted = answer
                    if citations:
                        formatted += "\n\n**Citations:** " + ", ".join(citations)

                    st.markdown(formatted)
                    st.session_state.messages.append({"role": "assistant", "content": formatted})
                except Exception as e:
                    err_msg = f"Error: {e}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg}) 