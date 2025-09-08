import streamlit as st
import skimage as ski
from ProjectFunctions import Tools
from ProjectFunctions import ColorSplit
from ProjectFunctions import RowColumnTransform
from ProjectFunctions import ChaoticSequence
from ProjectFunctions import CellularAutomata
from ProjectFunctions import ResultsAndDiscussions
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import io

st.set_page_config(page_title="CMSC 190", layout="wide")
with st.sidebar:
  st.title("`CMSC 190 Notebook`")
  st.write("A.J.N.T")

research_paper = "A Color Image Encryption Scheme Utilizing a Logistic-Sine Chaotic Map and Cellular Automata"

st.title(research_paper)
st.write('By: Shiji Sun, Wenzhong Yang, Yabo Yin, Xiaodan Tian, Guanghan Li, Xiangxin Deng')
st.markdown("---")

JellyFilePath = "static/Jelly.tiff"
Jelly = ski.io.imread(JellyFilePath)

st.subheader("Channel Split")
with st.expander("View Details"):
  col1, col2 = st.columns(2)
  with col1:
    st.info("This is using the normal channel split method.")
    st.code('''
def getChannels(image):
  redChannel = image[:, :, 0]
  greenChannel = image[:, :, 1]
  blueChannel = image[:, :, 2]
  return redChannel, greenChannel, blueChannel''', language="python")
    red_image, green_image, blue_image = ColorSplit.getChannels(Jelly)
    original_column, red_column, green_column, blue_column = st.columns(4)
    with original_column:
      st.image(Jelly, caption="Original Image", use_container_width=True)
    with red_column:
      st.image(red_image, caption="Red Channel", use_container_width=True)
    with green_column:
      st.image(green_image, caption="Green Channel", use_container_width=True)
    with blue_column:
      st.image(blue_image, caption="Blue Channel", use_container_width=True)
    red_colored, green_colored, blue_colored = Tools.channelsAsImages(red_image, green_image, blue_image, Jelly)
    red_colored_col, green_colored_col, blue_colored_col = st.columns(3)
    with red_colored_col:
      st.image(red_colored, caption="Red Channel (Colored)", use_container_width=True)
    with green_colored_col:
      st.image(green_colored, caption="Green Channel (Colored)", use_container_width=True)
    with blue_colored_col:
      st.image(blue_colored, caption="Blue Channel (Colored)", use_container_width=True)
  with col2:
    st.info("This is using the inverted channel split method.")
    st.code('''
def getInvertedChannels(image):
  redChannel = 255 - image[:, :, 0]
  greenChannel = 255 - image[:, :, 1]
  blueChannel = 255 - image[:, :, 2]
  return redChannel, greenChannel, blueChannel''', language="python")
    red_image, green_image, blue_image = ColorSplit.getChannels(Jelly)
    original_column, red_column, green_column, blue_column = st.columns(4)
    with original_column:
      st.image(Jelly, caption="Original Image", use_container_width=True)
    with red_column:
      st.image(red_image, caption="Red Channel", use_container_width=True)
    with green_column:
      st.image(green_image, caption="Green Channel", use_container_width=True)
    with blue_column:
      st.image(blue_image, caption="Blue Channel", use_container_width=True)
    red_colored, green_colored, blue_colored = Tools.channelsAsImages(red_image, green_image, blue_image, Jelly)
    red_colored_col, green_colored_col, blue_colored_col = st.columns(3)
    with red_colored_col:
      st.image(red_colored, caption="Red Channel (Colored)", use_container_width=True)
    with green_colored_col:
      st.image(green_colored, caption="Green Channel (Colored)", use_container_width=True)
    with blue_colored_col:
      st.image(blue_colored, caption="Blue Channel (Colored)", use_container_width=True)
  st.warning("Using the inverted channel split method is closer than with the reference paper **BUT** we continue with the normal channel split.")

st.subheader("Row and Column Transformations")
with st.expander("View Details"):
  st.write('''
  _For an encrypted image, we split it into a number of channels to perform different row and column transformations on it respectively.
  **The transformation starts with the second column.**_''')

  st.write('''
  Step 1: The odd columns of the G channel and the reverse order of the R channel are inverted. Every third column in
  the B channel is inverted.''')
  st.code('''def stepOne(red, green, blue):
  size = red.shape[0]
  
  # the odd columns of G channel are inverted
  for col in range(1, size):
    if col%2 == 1:
      green[:, col] = 255 - green[:, col]

  # the reverse order of the R channel are inverted
  for row in range(1, size):
    if row%2 == 1:
      red[row, :] = 255 - red[row, :]
  
  # every third column in the B channel is inverted
  for col in range(1, size):
    if (col-1)%3 == 0:
      blue[:, col] = 255 - blue[:, col]

  return red, green, blue''')
  red, green, blue = ColorSplit.getChannels(Jelly)
  red, green, blue = RowColumnTransform.stepOne(red, green, blue)
  red_image, green_image, blue_image = Tools.channelsAsImages(red, green, blue, Jelly)
  col1, col2, col3 = st.columns(3)
  with col1:
    st.image(red_image, caption="Red Channel (after step 1)", use_container_width=True)
  with col2:
    st.image(green_image, caption="Green Channel (after step 1)", use_container_width=True)
  with col3:
    st.image(blue_image, caption="Blue Channel (after step 1)", use_container_width=True)

  st.write('''
    Step 2: The R and G channels are switched in every column. The R and B channels 
    are switched in every third column. G and B channels switched in every 5 column.''')
  st.code('''def stepTwo(red, green, blue):
  size = red.shape[0]

  # The R and G are switched in every column
  for col in range(1, size):
    red[:, col], green[:, col] = green[:, col], red[:, col]

  # The R and B channels are switched in every third column
  for col in range(1, size):
    if (col-1)%3 == 0:
      red[:, col], blue[:, col] = blue[:, col], red[:, col]

  # The G and B channels are switched in every fifth column
  for col in range(1, size):
    if (col-1)%5 == 0:
      green[:, col], blue[:, col] = blue[:, col], green[:, col]

  return red, green, blue''')
  red, green, blue = RowColumnTransform.stepTwo(red, green, blue)
  red_image, green_image, blue_image = Tools.channelsAsImages(red, green, blue, Jelly)
  col1, col2, col3 = st.columns(3)
  with col1:
    st.image(red_image, caption="Red Channel (after step 2)", use_container_width=True)
  with col2:
    st.image(green_image, caption="Green Channel (after step 2)", use_container_width=True)
  with col3:
    st.image(blue_image, caption="Blue Channel (after step 2)", use_container_width=True)

  st.write('''Step 3: An inversion operation is performed for each channel.''')
  st.code('''def stepThree(red, green, blue):
  red = 255 - red
  green = 255 - green
  blue = 255 - blue
  return red, green, blue''')
  red, green, blue = RowColumnTransform.stepThree(red, green, blue)
  red_image, green_image, blue_image = Tools.channelsAsImages(red, green, blue, Jelly)
  col1, col2, col3 = st.columns(3)
  with col1:
    st.image(red_image, caption="Red Channel (after step 3)", use_container_width=True)
  with col2:
    st.image(green_image, caption="Green Channel (after step 3)", use_container_width=True)
  with col3:
    st.image(blue_image, caption="Blue Channel (after step 3)", use_container_width=True)

  st.write('''Step 4: A cyclic shift operation is performed on each row of the R channel, every three rows of the B channel, 
and every five rows of the G channel.''')
  st.code('''def stepFour(red, green, blue):
  size = red.shape[0]

  # A cyclic shift is performed on each row of the R channel
  for row in range(size):
    red[row, :] = np.roll(red[row, :], row)

  # A cyclic shift is performed on every three rows of the B channel
  for row in range(0, size, 3):
    blue[row, :] = np.roll(blue[row, :], row)

  # A cyclic shift is performed on every five rows of the G channel
  for row in range(0, size, 5):
    green[row, :] = np.roll(green[row, :], row)

  return red, green, blue''')

  red, green, blue = RowColumnTransform.stepFour(red, green, blue)
  red_image, green_image, blue_image = Tools.channelsAsImages(red, green, blue, Jelly)
  col1, col2, col3 = st.columns(3)
  with col1:
    st.image(red_image, caption="Red Channel (after step 4)", use_container_width=True)
  with col2:
    st.image(green_image, caption="Green Channel (after step 4)", use_container_width=True)
  with col3:
    st.image(blue_image, caption="Blue Channel (after step 4)", use_container_width=True)

  st.write('''Step 5: The row-column transformation is completed.''')
  transformedImage = Tools.mergeChannels(red, green, blue)
  red_image, green_image, blue_image = Tools.channelsAsImages(red, green, blue, Jelly)
  col1, col2, col3 = st.columns(3)
  with col1:
    st.image(red_image, caption="Red Channel (after step 5)", use_container_width=True)
  with col2:
    st.image(green_image, caption="Green Channel (after step 5)", use_container_width=True)
  with col3:
    st.image(blue_image, caption="Blue Channel (after step 5)", use_container_width=True)

st.subheader("Generating chaotic sequences")
with st.expander("View Details"):
  size = transformedImage.shape[0] * transformedImage.shape[1] * 3
  LCM_CM = ChaoticSequence.getChaoticImage("LCM", size, transformedImage.shape[0])
  SCM_CM = ChaoticSequence.getChaoticImage("SCM", size, transformedImage.shape[0])
  LSCM_CM = ChaoticSequence.getChaoticImage("LSCM", size, transformedImage.shape[0])
  col1, col2, col3 = st.columns(3)
  with col1:
    st.latex(r'''x_{n+1} = r\times x_n \times (1 - x_n)''')
    st.code('''def getNext_LCM(lastValue):
  r = 4
  return r * lastValue * (1 - lastValue)''')
    st.image(LCM_CM, caption="Logistic Chaotic Map", use_container_width=True)
  with col2:
    st.latex(r'''x_{n+1} = \frac{r}{4} \times sin(\pi \times x_n)''')
    st.code('''def getNext_SCM(lastValue):
  r = 4
  return r/4*np.sin(np.pi * lastValue)''')
    st.image(SCM_CM, caption="Sine Chaotic Map", use_container_width=True)
  with col3:
    st.latex(r'''x_{n+1} = sin(r \times \pi \times (1 - x_n) \times b \times x_n)''')
    st.code('''def get_Next_LSCM(lastValue):
  r = 4
  b = 4
  return np.sin(r*np.pi*(1-lastValue)*b*lastValue)''')
    st.image(LSCM_CM, caption="Logistic Sine Chaotic Map", use_container_width=True)

  chaoticXOR_image = ChaoticSequence.XOR_images(transformedImage.copy(), LSCM_CM)

  st.image(chaoticXOR_image, caption="XOR between transformed image and LSCM chaotic map")

st.subheader("Cellular automata process")
with st.expander("View Details"):
  CA_image = CellularAutomata.generateCAImage(chaoticXOR_image.copy())
  col1, col2 = st.columns(2)
  with col1:
    st.image(chaoticXOR_image, caption="Image before Cellular Automata", use_container_width=True)
  with col2:
    st.image(CA_image, caption="Cellular Automata Image", use_container_width=True)

st.subheader("Decryption process")
decrypted_image = CA_image.copy()
with st.expander("View Details"):
  reversed_CA_image = CellularAutomata.generateCAImage(decrypted_image.copy())
  st.image(reversed_CA_image, caption="After reversing Cellular Automata")
  reversed_XOR_image = ChaoticSequence.XOR_images(reversed_CA_image, LSCM_CM)
  st.write("After reversing XOR with LSCM chaotic, is it equal to the transformed image?(checked by: `(reversed_XOR_image == transformedImage).all()`): " + str((reversed_XOR_image == transformedImage).all()))
  st.image(reversed_XOR_image, caption="After reversing XOR with LSCM chaotic")
  red, blue, green = reversed_XOR_image[:, :, 0], reversed_XOR_image[:, :, 2], reversed_XOR_image[:, :, 1]
  red_image, green_image, blue_image = Tools.channelsAsImages(red, green, blue, Jelly)
  col1, col2, col3 = st.columns(3)
  with col1:
    st.image(red_image, caption="Red Channel (before reversing row-column transform)", use_container_width=True)
  with col2:
    st.image(green_image, caption="Green Channel (before reversing row-column transform)", use_container_width=True)
  with col3:
    st.image(blue_image, caption="Blue Channel (before reversing row-column transform)", use_container_width=True)

  red, green, blue = RowColumnTransform.reverseStepFour(red, green, blue)
  red_image, green_image, blue_image = Tools.channelsAsImages(red, green, blue, Jelly)
  col1, col2, col3 = st.columns(3)
  with col1:
    st.image(red_image, caption="Red Channel (after reversing step 4)", use_container_width=True)
  with col2:
    st.image(green_image, caption="Green Channel (after reversing step 4)", use_container_width=True)
  with col3:
    st.image(blue_image, caption="Blue Channel (after reversing step 4)", use_container_width=True)

  red, green, blue = RowColumnTransform.stepThree(red, green, blue)
  red_image, green_image, blue_image = Tools.channelsAsImages(red, green, blue, Jelly)
  col1, col2, col3 = st.columns(3)
  with col1:
    st.image(red_image, caption="Red Channel (after reversing step 3)", use_container_width=True)
  with col2:
    st.image(green_image, caption="Green Channel (after reversing step 3)", use_container_width=True)
  with col3:
    st.image(blue_image, caption="Blue Channel (after reversing step 3)", use_container_width=True)
  red, green, blue = RowColumnTransform.reverseStepTwo(red, green, blue)
  red_image, green_image, blue_image = Tools.channelsAsImages(red, green, blue, Jelly)
  col1, col2, col3 = st.columns(3)
  with col1:
    st.image(red_image, caption="Red Channel (after reversing step 2)", use_container_width=True)
  with col2:
    st.image(green_image, caption="Green Channel (after reversing step 2)", use_container_width=True)
  with col3:
    st.image(blue_image, caption="Blue Channel (after reversing step 2)", use_container_width=True)
  red, green, blue = RowColumnTransform.reverseStepOne(red, green, blue)
  red_image, green_image, blue_image = Tools.channelsAsImages(red, green, blue, Jelly)
  col1, col2, col3 = st.columns(3)
  with col1:
    st.image(red_image, caption="Red Channel (after reversing step 1)", use_container_width=True)
  with col2:
    st.image(green_image, caption="Green Channel (after reversing step 1)", use_container_width=True)
  with col3:
    st.image(blue_image, caption="Blue Channel (after reversing step 1)", use_container_width=True)
  decrypted_image = Tools.mergeChannels(red, green, blue)
  st.image(decrypted_image, caption="Decrypted Image")

def plot_single_histogram(image, title="Histogram Analysis"):
  gray = ski.color.rgb2gray(image)*255
  hist, _ = np.histogram(gray, bins=256, range=[0,256])
  df = pd.DataFrame({'Pixel Value': np.arange(256), 'Frequency': hist})
  st.write(f"**{title}**")
  st.bar_chart(df.set_index('Pixel Value'))

st.subheader("Final Results")
col1, col2, col3 = st.columns(3)
with col1:
  st.image(ski.io.imread(JellyFilePath), caption="Original Image", use_container_width=True)
with col2:
  st.image(CA_image, caption="Encrypted Image", use_container_width=True)
with col3:
  st.image(decrypted_image, caption="Decrypted Image", use_container_width=True)



st.write("---")
st.header("Results and Discussions")
st.subheader("Histogram Analysis")
col1, col2, col3 = st.columns(3)  
with col1:
  plot_single_histogram(ski.io.imread(JellyFilePath), "Original Image Histogram")
with col2:
  plot_single_histogram(CA_image, "Encrypted Image Histogram")
with col3:
  plot_single_histogram(decrypted_image, "Decrypted Image Histogram")

