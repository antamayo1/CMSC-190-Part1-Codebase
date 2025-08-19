import streamlit as st
import skimage as ski
import numpy as np
from functions.functions import get_RGB_channels

st.set_page_config(page_title="CMSC 190 - Part 1", layout="wide")

st.title('CMSC 190 - Part 1 Journal')
st.write('By: **Aaron John N. Tamayo**')
st.markdown("---")
st.subheader("Tuesday, 19 August 2025")
st.markdown('''
Approved reference paper titled `A Color Image Encryption Scheme Utilizing a Logistic-Sine Chaotic Map and Cellular Automata`.
Currently reading the introduction and preliminaries of the paper but I have tried to start the methodology section
by downloading the images in the paper and applying the channel split with just the image named `jelly.tiff`.
''')
code_1, code_2 = st.columns(2)
with code_1:
  st.write("**Initial Implementation**")
  st.code('''
  def get_RGB_channels(image):
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    return red_channel, green_channel, blue_channel
  ''', language='python')
with code_2:
  st.write("**Inverted Implementation**")
  st.code('''
  def get_RGB_channels(image):
    red_channel = 255 - image[:, :, 0]
    green_channel = 255 - image[:, :, 1]
    blue_channel = 255 - image[:, :, 2]
    return red_channel, green_channel, blue_channel
  ''', language='python')
st.markdown('''
The initial output does not seem to have similar output as the reference paper. When I tried inverting the channels
the output was still different but closer to the expected result. Need to investigate the statement "_The R, G, and
B are transformed with different ranks to increase deciphering difficulty_" or is this statement about the row-wise
and column-wise transformations before encryption?
''')
st.markdown('''
**GDOCS UPDATE**:
> Added the download links for the used images.
''')
st.write("---")
st.header('Notes')
st.subheader('Channel Split')
st.write('''
* A colored image is an M x N x 3 array, where M is the height, N is the width, and 3 represents the RGB channels.
* The channel split process separates the image into its individual R, G, and B components.
* The current implementation uses `skimage` to read the image into a numpy array.
''')