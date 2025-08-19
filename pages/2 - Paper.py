import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

pdf_path = "static/reference.pdf"

if "pdf_path" not in st.session_state:
  st.session_state["pdf_path"] = pdf_path

with st.spinner("Loading PDF..."):
  pdf_viewer(st.session_state["pdf_path"])