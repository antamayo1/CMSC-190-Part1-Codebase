import streamlit as st
import skimage as ski
import numpy as np
from functions.functions import get_RGB_channels

st.set_page_config(page_title="CMSC 190 - Part 1", layout="wide")

research_paper = "A Color Image Encryption Scheme Utilizing a Logistic-Sine Chaotic Map and Cellular Automata"

st.title(research_paper)
st.write('By: Shiji Sun, Wenzhong Yang, Yabo Yin, Xiaodan Tian, Guanghan Li, Xiangxin Deng')
st.markdown("---")

image_path = "static/Jelly.tiff"
image = ski.io.imread(image_path)

st.header("1. Channel Split")
st.code('''
def get_RGB_channels(image):
  red_channel = image[:, :, 0]
  green_channel = image[:, :, 1]
  blue_channel = image[:, :, 2]
  return red_channel, green_channel, blue_channel
''', language="python")
red_image, green_image, blue_image = get_RGB_channels(image)
original_column, red_column, green_column, blue_column = st.columns(4)
with original_column:
  st.image(image, caption="Original Image", use_container_width=True)
with red_column:
  st.image(red_image, caption="Red Channel", use_container_width=True)
with green_column:
  st.image(green_image, caption="Green Channel", use_container_width=True)
with blue_column:
  st.image(blue_image, caption="Blue Channel", use_container_width=True)