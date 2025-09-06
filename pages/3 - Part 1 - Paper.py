import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

st.set_page_config(page_title="CMSC 190", layout="wide")
with st.sidebar:
  st.title("`CMSC 190 Notebook`")
  st.write("A.J.N.T")

pdf_path = "static/reference.pdf"

if "pdf_path" not in st.session_state:
  st.session_state["pdf_path"] = pdf_path
  
with st.spinner("Loading PDF..."):
  pdf_viewer(st.session_state["pdf_path"])