import os
from pathlib import Path
from typing import List

import requests
import streamlit as st

API_URL = "http://localhost:8000"  # FastAPI backend base
UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"

st.set_page_config(page_title="HR Admin Dashboard", page_icon="🗂️")

st.title("🗂️ HR Onboarding Assistant – Admin Dashboard")

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

files: List[Path] = sorted(UPLOAD_DIR.glob("*"))

st.subheader("Uploaded HR Documents")
if not files:
    st.info("No HR documents have been uploaded yet.")
else:
    for file_path in files:
        cols = st.columns([6, 2])
        cols[0].markdown(f"**{file_path.name}**  ")
        if cols[1].button("Delete", key=str(file_path)):
            try:
                file_path.unlink(missing_ok=False)
                st.success(f"Deleted {file_path.name}")
                st.experimental_rerun()
            except Exception as exc:
                st.error(f"Failed to delete {file_path.name}: {exc}")

st.divider()

st.caption("Changes take effect immediately. Re-upload documents if needed.") 