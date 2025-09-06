import streamlit as st

st.set_page_config(page_title="CMSC 190", layout="wide")
with st.sidebar:
  st.title("`CMSC 190 Notebook`")
  st.write("A.J.N.T")

st.title("Welcome to my CMSC 190 Notebook", anchor=False)
st.write("This notebook contains my notes and code implementations for the CMSC 190 course.")
st.markdown("---")
st.subheader("Part 1 Reference Paper Title")
st.write("`A Color Image Encryption Scheme Utilizing a Logistic-Sine Chaotic Map and Cellular Automata`")
st.write("By: Shiji Sun, Wenzhong Yang, Yabo Yin, Xiaodan Tian, Guanghan Li, Xiangxin Deng")

st.page_link("pages/1 - Part 1 - Journal.py", label="Check my Journal and Notes")
st.page_link("pages/2 - Part 1 - Methodology.py", label="Check Replication of Methodology")
st.page_link("pages/3 - Part 1 - Paper.py", label="Check the Reference Paper")
st.markdown("---")
st.subheader("Part 2 Paper Title")
st.write("`SOON`")