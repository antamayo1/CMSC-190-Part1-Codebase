import streamlit as st
import skimage as ski
from ProjectFunctions import Tools
from ProjectFunctions import ColorSplit
from ProjectFunctions import RowColumnTransform
from ProjectFunctions import ChaoticSequence
from ProjectFunctions import CellularAutomata
from ProjectFunctions import ResultsAndDiscussions
import numpy as np
import pandas as pd

st.set_page_config(page_title="CMSC 190", layout="wide")
with st.sidebar:
  st.title("`CMSC 190 Notebook`")
  st.write("A.J.N.T")

research_paper = "A Color Image Encryption Scheme Utilizing a Logistic-Sine Chaotic Map and Cellular Automata"

st.title(research_paper)
st.write('By: Shiji Sun, Wenzhong Yang, Yabo Yin, Xiaodan Tian, Guanghan Li, Xiangxin Deng')

# Image Selection Section
st.markdown("---")

st.subheader("Image Selection") 
images = {
  "Baboon": "static/Baboon.tiff",
  "House": "static/House.tiff",
  "Jelly": "static/Jelly.tiff",
}

column1, column2 = st.columns(2)
with column1:
  st.session_state.originalImage = st.file_uploader("Upload your own image (optional)", type=["png", "jpg", "jpeg", "tiff"])
  st.caption("You can upload your own image.")
  st.session_state.selectedImage = st.selectbox("Select an image", list(images.keys()), index=2)
  st.session_state.imagePath = images[st.session_state.selectedImage]
  st.caption("The images that are in the selectbox are from the reference paper.")
  if st.session_state.originalImage is None:
    st.session_state.originalImage = ski.io.imread(st.session_state.imagePath)[:,:,:3]
  else:
    st.session_state.originalImage = ski.io.imread(st.session_state.originalImage)[:,:,:3]
  st.write("**Image selected:** " + f'`{st.session_state.imagePath}`')
  st.write("**Image shape:** " + f'`{str(st.session_state.originalImage.shape)}`')
with column2:
  with st.container(horizontal_alignment='center', horizontal=True):
    st.image(st.session_state.originalImage, caption="Selected Image")

# Channel Split Section
st.markdown("---")

st.subheader("Channel Split")
st.write('This the step where we split the image into individual color channels `R`, `G` and `B`. Note that the shape of the input image is $(h, w, 3)$ where $3$ represents the channels.')
col1, col2 = st.columns(2)
with col1:
  st.info("This is using the normal channel split method.", icon="ℹ️")
  st.code('''
def getChannels(image):
  redChannel = image[:, :, 0]
  greenChannel = image[:, :, 1]
  blueChannel = image[:, :, 2]
  return redChannel, greenChannel, blueChannel''', language="python")
  red_image, green_image, blue_image = ColorSplit.getChannels(st.session_state.originalImage.copy())
  original_column, red_column, green_column, blue_column = st.columns(4)
  with original_column:
    st.image(st.session_state.originalImage, caption="Original Image", use_container_width=True)
  with red_column:
    st.image(red_image, caption="Red Channel", use_container_width=True)
  with green_column:
    st.image(green_image, caption="Green Channel", use_container_width=True)
  with blue_column:
    st.image(blue_image, caption="Blue Channel", use_container_width=True)
  red_colored, green_colored, blue_colored = Tools.channelsAsImages(red_image.copy(), green_image.copy(), blue_image.copy(), st.session_state.originalImage.copy())
  red_colored_col, green_colored_col, blue_colored_col = st.columns(3)
  with red_colored_col:
    st.image(red_colored, caption="Red Channel (Colored)", use_container_width=True)
  with green_colored_col:
    st.image(green_colored, caption="Green Channel (Colored)", use_container_width=True)
  with blue_colored_col:
    st.image(blue_colored, caption="Blue Channel (Colored)", use_container_width=True)
with col2:
  st.info("This is using the inverted channel split method.", icon="ℹ️")
  st.code('''
def getInvertedChannels(image):
  redChannel = 255 - image[:, :, 0]
  greenChannel = 255 - image[:, :, 1]
  blueChannel = 255 - image[:, :, 2]
  return redChannel, greenChannel, blueChannel''', language="python")
  red_image, green_image, blue_image = ColorSplit.getInvertedChannels(st.session_state.originalImage.copy())
  original_column, red_column, green_column, blue_column = st.columns(4)
  with original_column:
    st.image(st.session_state.originalImage, caption="Original Image", use_container_width=True)
  with red_column:
    st.image(red_image, caption="Red Channel", use_container_width=True)
  with green_column:
    st.image(green_image, caption="Green Channel", use_container_width=True)
  with blue_column:
    st.image(blue_image, caption="Blue Channel", use_container_width=True)
  red_colored, green_colored, blue_colored = Tools.channelsAsImages(red_image.copy(), green_image.copy(), blue_image.copy(), st.session_state.originalImage.copy())
  red_colored_col, green_colored_col, blue_colored_col = st.columns(3)
  with red_colored_col:
    st.image(red_colored, caption="Red Channel (Colored)", use_container_width=True)
  with green_colored_col:
    st.image(green_colored, caption="Green Channel (Colored)", use_container_width=True)
  with blue_colored_col:
    st.image(blue_colored, caption="Blue Channel (Colored)", use_container_width=True)
st.warning("Using the inverted channel split method is closer than with the reference paper **BUT** we continue with the normal channel split.", icon="⚠️")

# Row and Column Transformations Section
st.markdown("---")
st.subheader("Row and Column Transformations")
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
  for row in range(size):
    if row%2 == 1:
      red[1:row, :] = 255 - red[1:row, :]

  # every third column in the B channel is inverted
  for col in range(1, size):
    if (col-1)%3 == 0:
      blue[:, col] = 255 - blue[:, col]

  return red, green, blue''')
red, green, blue = ColorSplit.getChannels(st.session_state.originalImage.copy())
red, green, blue = RowColumnTransform.stepOne(red.copy(), green.copy(), blue.copy())
red_image, green_image, blue_image = Tools.channelsAsImages(red.copy(), green.copy(), blue.copy(), st.session_state.originalImage.copy())
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
red, green, blue = RowColumnTransform.stepTwo(red.copy(), green.copy(), blue.copy())
red_image, green_image, blue_image = Tools.channelsAsImages(red.copy(), green.copy(), blue.copy(), st.session_state.originalImage.copy())
col1, col2, col3 = st.columns(3)
with col1:
  st.image(red_image, caption="Red Channel (after step 2)", use_container_width=True)
with col2:
  st.image(green_image, caption="Green Channel (after step 2)", use_container_width=True)
with col3:
  st.image(blue_image, caption="Blue Channel (after step 2)", use_container_width=True)

st.write('''Step 3: An inversion operation is performed for each channel.''')
st.code('''def stepThree(red, green, blue):
  # Inversion operation for each channel
  red[1:] = 255 - red[1:]
  green[1:] = 255 - green[1:]
  blue[1:] = 255 - blue[1:]
  return red, green, blue''')
red, green, blue = RowColumnTransform.stepThree(red.copy(), green.copy(), blue.copy())
red_image, green_image, blue_image = Tools.channelsAsImages(red.copy(), green.copy(), blue.copy(), st.session_state.originalImage.copy())
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
  for row in range(1, size):
    red[row, :] = np.roll(red[row, :], row)

  # A cyclic shift is performed on every three rows of the B channel
  for row in range(1, size, 3):
    blue[row, :] = np.roll(blue[row, :], row)

  # A cyclic shift is performed on every five rows of the G channel
  for row in range(1, size, 5):
    green[row, :] = np.roll(green[row, :], row)

  return red, green, blue''')

red, green, blue = RowColumnTransform.stepFour(red.copy(), green.copy(), blue.copy())
red_image, green_image, blue_image = Tools.channelsAsImages(red.copy(), green.copy(), blue.copy(), st.session_state.originalImage.copy())
col1, col2, col3 = st.columns(3)
with col1:
  st.image(red_image, caption="Red Channel (after step 4)", use_container_width=True)
with col2:
  st.image(green_image, caption="Green Channel (after step 4)", use_container_width=True)
with col3:
  st.image(blue_image, caption="Blue Channel (after step 4)", use_container_width=True)

st.write('''Step 5: The row-column transformation is completed.''')
st.session_state.transformedImage = Tools.mergeChannels(red.copy(), green.copy(), blue.copy())
with st.container(horizontal_alignment='center', horizontal=True):
  st.image(st.session_state.transformedImage, caption="Transformed Image after Row-Column Transformations")
st.warning("Transformed image was not shown in the reference paper.", icon="⚠️")

# Chaotic Sequence Generation Section
st.markdown("---")

st.subheader("Generating chaotic sequences")
st.write('These are differenct chaotic maps that were mentioned in the reference paper, and **Logistic Sine Chaotic Map** is introduced and used.')
size = st.session_state.originalImage.shape[0] * st.session_state.originalImage.shape[1] * 3
st.session_state.LCM_CM = ChaoticSequence.getChaoticImage("LCM", size, st.session_state.originalImage.shape[0])
st.session_state.SCM_CM = ChaoticSequence.getChaoticImage("SCM", size, st.session_state.originalImage.shape[0])
st.session_state.LSCM_CM = ChaoticSequence.getChaoticImage("LSCM", size, st.session_state.originalImage.shape[0])
col1, col2, col3 = st.columns(3)
with col1:
  st.latex(r'''x_{n+1} = r\times x_n \times (1 - x_n)''')
  st.code('''def getNext_LCM(lastValue):
  r = 4
  return r * lastValue * (1 - lastValue)''')
  st.image(st.session_state.LCM_CM, caption="Logistic Chaotic Map", use_container_width=True)
with col2:
  st.latex(r'''x_{n+1} = \frac{r}{4} \times sin(\pi \times x_n)''')
  st.code('''def getNext_SCM(lastValue):
  r = 4
  return r/4*np.sin(np.pi * lastValue)''')
  st.image(st.session_state.SCM_CM, caption="Sine Chaotic Map", use_container_width=True)
with col3:
  st.latex(r'''x_{n+1} = sin(r \times \pi \times (1 - x_n) \times b \times x_n)''')
  st.code('''def get_Next_LSCM(lastValue):
  r = 4
  b = 4
  return np.sin(r*np.pi*(1-lastValue)*b*lastValue)''')
  st.image(st.session_state.LSCM_CM, caption="Logistic Sine Chaotic Map", use_container_width=True)
st.scatter_chart(ChaoticSequence.getSequenceOnly("LSCM", 5000))
with st.container(horizontal_alignment='center', horizontal=True):
  st.caption("LSCM chaotic sequence scatter plot", width="content")
st.warning("The range of the LCSM is said to be $(0,1)$ but it is a $\\sin$ function which means the range should be $[-1,1]$.", icon="⚠️")
st.write('We need to **XOR** the chaotic map with the transformed image.')
st.code('''def XOR_images(imageA, imageB):
  return np.bitwise_xor(imageA, imageB)''')
st.write("We can utilize the `numpy.bitwise_xor` function which takes values of decimal and converts them to binary then performs XOR operation.")
st.session_state.chaoticXOR_image = ChaoticSequence.XOR_images(st.session_state.transformedImage.copy(), st.session_state.LSCM_CM.copy())
column1, column2, column3 = st.columns(3)
with column1:
  st.image(st.session_state.transformedImage, caption="Transformed Image", use_container_width=True)
with column2:
  st.image(st.session_state.LSCM_CM, caption="LSCM Chaotic Map", use_container_width=True)
with column3:
  st.image(st.session_state.chaoticXOR_image, caption="XOR between transformed image and LSCM chaotic map", use_container_width=True)

# Cellular Automata Section
st.markdown("---")

st.subheader("Cellular automata process")
st.write('The Ceullular Automata rules have 2 main functions')
column1, column2 = st.columns(2)
with column1:
  st.code('''def applyXOR(pixel):
  new_pixel = pixel
  new_pixel = list(new_pixel)
  for idx in range(4):
    if pixel[idx] == pixel[(idx+1)]:
      new_pixel[idx] = '0'
    else:
      new_pixel[idx] = '1'
  return ''.join(new_pixel)''', language="python")
  st.info("This function checks the first 5 bits if the old pixel and determine the first 4 bits of the new pixel.", icon="ℹ️")
with column2:
  st.code('''def applyinvert(pixel):
  new_pixel = pixel
  new_pixel = list(new_pixel)
  for idx in range(4, 8):
    if pixel[idx] == '0':
      new_pixel[idx] = '1'
    else:
      new_pixel[idx] = '0'
  return ''.join(new_pixel)''', language="python")
  st.info("This function inverts the last 4 bits of the pixel.", icon="ℹ️")
st.write('As per the reference paper, these two functions are applied iteratively for 4 rounds which is implemented in the following code.')
st.code('''def applyCATransform(pixel):
  old_pixel = pixel
  for idx in range(4):
    XOR_pixel = CellularAutomata.applyXOR(old_pixel)
    inverted_pixel = CellularAutomata.applyinvert(old_pixel)
    old_pixel = XOR_pixel[:4] + inverted_pixel[4:]
  return int(old_pixel, 2)

def generateCAImage(image):
  orig = image.copy()
  for row in range(image.shape[0]):
    for col in range(image.shape[1]):
      for channel in range(image.shape[2]):
        binary_pixel = np.binary_repr(image[row, col, channel], width=8)
        image[row, col, channel] = CellularAutomata.applyCATransform(binary_pixel)
  return image''', language="python")
st.write('To check the results of the Cellular Automata process, I have added a number input and determine the result.')
column1, column2 = st.columns(2)
with column1:
  test_number = st.number_input("Enter a number between 0-255 to see the Cellular Automata process", min_value=0, max_value=255, value=153)
with column2:
  binary_pixel = np.binary_repr(test_number, width=8)
  ca_result = CellularAutomata.applyCATransform(binary_pixel)
  st.number_input("Result after Cellular Automata process", value=ca_result, disabled=True)
st.session_state.encryptedImage = CellularAutomata.generateCAImage(st.session_state.chaoticXOR_image.copy())
col1, col2 = st.columns(2)
with col1:
  st.image(st.session_state.chaoticXOR_image, caption="Image before Cellular Automata", use_container_width=True)
with col2:
  st.image(st.session_state.encryptedImage, caption="Encrypted Image", use_container_width=True)

st.markdown("---")

st.subheader("Decryption process")
st.write("For the decryption process, we reverse all the steps that were done in the encryption process. Starting from the `Cellular Automata` and `XOR` with the `LCSM chaotic map`.")
column1, column2, column3 = st.columns(3)
with column1:
  st.image(st.session_state.encryptedImage, caption="Encrypted Image", use_container_width=True)
with column2:
  st.session_state.decrypt_CA = CellularAutomata.generateCAImage(st.session_state.encryptedImage)
  st.image(st.session_state.decrypt_CA, caption="After reversing Cellular Automata", use_container_width=True)
with column3:
  st.session_state.decrypt_XOR = ChaoticSequence.XOR_images(st.session_state.decrypt_CA, st.session_state.LSCM_CM)
  st.image(st.session_state.decrypt_XOR, caption="After reversing XOR with LSCM chaotic", use_container_width=True)
st.write("After reversing the Cellular Automata and XOR with LSCM chaotic map, we can now reverse the Row-Column transformations where we split the image again.")

red, green, blue = ColorSplit.getChannels(st.session_state.decrypt_XOR.copy())
red_image, green_image, blue_image = Tools.channelsAsImages(red.copy(), green.copy(), blue.copy(), st.session_state.originalImage.copy())
col1, col2, col3 = st.columns(3)
with col1:
  st.image(red_image, caption="Red Channel of Encrypted Image", use_container_width=True)
with col2:
  st.image(green_image, caption="Green Channel of Encrypted Image", use_container_width=True)
with col3:
  st.image(blue_image, caption="Blue Channel of Encrypted Image", use_container_width=True)

red, green, blue = RowColumnTransform.reverseStepFour(red.copy(), green.copy(), blue.copy())
red_image, green_image, blue_image = Tools.channelsAsImages(red.copy(), green.copy(), blue.copy(), st.session_state.originalImage.copy())
col1, col2, col3 = st.columns(3)
with col1:
  st.image(red_image, caption="Red Channel (after reversing step 4)", use_container_width=True)
with col2:
  st.image(green_image, caption="Green Channel (after reversing step 4)", use_container_width=True)
with col3:
  st.image(blue_image, caption="Blue Channel (after reversing step 4)", use_container_width=True)

red, green, blue = RowColumnTransform.stepThree(red.copy(), green.copy(), blue.copy())
red_image, green_image, blue_image = Tools.channelsAsImages(red.copy(), green.copy(), blue.copy(), st.session_state.originalImage.copy())
col1, col2, col3 = st.columns(3)
with col1:
  st.image(red_image, caption="Red Channel (after reversing step 3)", use_container_width=True)
with col2:
  st.image(green_image, caption="Green Channel (after reversing step 3)", use_container_width=True)
with col3:
  st.image(blue_image, caption="Blue Channel (after reversing step 3)", use_container_width=True)

red, green, blue = RowColumnTransform.reverseStepTwo(red.copy(), green.copy(), blue.copy())
red_image, green_image, blue_image = Tools.channelsAsImages(red.copy(), green.copy(), blue.copy(), st.session_state.originalImage.copy())
col1, col2, col3 = st.columns(3)
with col1:
  st.image(red_image, caption="Red Channel (after reversing step 2)", use_container_width=True)
with col2:
  st.image(green_image, caption="Green Channel (after reversing step 2)", use_container_width=True)
with col3:
  st.image(blue_image, caption="Blue Channel (after reversing step 2)", use_container_width=True)

red, green, blue = RowColumnTransform.reverseStepOne(red.copy(), green.copy(), blue.copy())
red_image, green_image, blue_image = Tools.channelsAsImages(red.copy(), green.copy(), blue.copy(), st.session_state.originalImage.copy())
col1, col2, col3 = st.columns(3)
with col1:
  st.image(red_image, caption="Red Channel (after reversing step 1)", use_container_width=True)
with col2:
  st.image(green_image, caption="Green Channel (after reversing step 1)", use_container_width=True)
with col3:
  st.image(blue_image, caption="Blue Channel (after reversing step 1)", use_container_width=True)

st.session_state.decryptedImage = Tools.mergeChannels(red.copy(), green.copy(), blue.copy())
with st.container(horizontal_alignment='center', horizontal=True):
  st.image(st.session_state.decryptedImage, caption="Decrypted Image")

def plot_single_histogram(image, title="Histogram Analysis"):
  gray = ski.color.rgb2gray(image)*255
  hist, _ = np.histogram(gray, bins=256, range=[0,256])
  df = pd.DataFrame({'Pixel Value': np.arange(256), 'Frequency': hist})
  st.write(f"**{title}**")
  st.bar_chart(df.set_index('Pixel Value'))

st.subheader("Final Results")
col1, col2, col3 = st.columns(3)
with col1:
  st.image(st.session_state.originalImage, caption="Original Image", use_container_width=True)
with col2:
  st.image(st.session_state.encryptedImage, caption="Encrypted Image", use_container_width=True)
with col3:
  st.image(st.session_state.decryptedImage, caption="Decrypted Image", use_container_width=True)

st.write("---")
st.header("Results and Discussions")
st.subheader("Histogram Analysis")
col1, col2, col3 = st.columns(3)  
with col1:
  plot_single_histogram(st.session_state.originalImage, "Original Image Histogram")
with col2:
  plot_single_histogram(st.session_state.encryptedImage, "Encrypted Image Histogram")
with col3:
  plot_single_histogram(st.session_state.decryptedImage, "Decrypted Image Histogram")
st.info("The histogram of the original image and the decrypted image are very similar, we can say that the decryption is successful. As for the encrypted image, the histogram is a little bit normally distributed which is not the same as the reference paper.", icon="ℹ️")
st.subheader("Correlation Analysis")
st.write('_**Original Image Correlation Results**_')
orig_red_horizontal_pairs = ResultsAndDiscussions.getHorizontalPairs(st.session_state.originalImage[:,:,0])
orig_green_horizontal_pairs = ResultsAndDiscussions.getHorizontalPairs(st.session_state.originalImage[:,:,1])
orig_blue_horizontal_pairs = ResultsAndDiscussions.getHorizontalPairs(st.session_state.originalImage[:,:,2])

orig_red_vertical_pairs = ResultsAndDiscussions.getVerticalPairs(st.session_state.originalImage[:,:,0])
orig_green_vertical_pairs = ResultsAndDiscussions.getVerticalPairs(st.session_state.originalImage[:,:,1])
orig_blue_vertical_pairs = ResultsAndDiscussions.getVerticalPairs(st.session_state.originalImage[:,:,2])

orig_red_diagonal_pairs = ResultsAndDiscussions.getDiagonalPairs(st.session_state.originalImage[:,:,0])
orig_green_diagonal_pairs = ResultsAndDiscussions.getDiagonalPairs(st.session_state.originalImage[:,:,1])
orig_blue_diagonal_pairs = ResultsAndDiscussions.getDiagonalPairs(st.session_state.originalImage[:,:,2])

orig_red_horizontal_df = pd.DataFrame({
  'Pixel Value': orig_red_horizontal_pairs[:,0],
  'Neighbor Value': orig_red_horizontal_pairs[:,1]
})

orig_green_horizontal_df = pd.DataFrame({
  'Pixel Value': orig_green_horizontal_pairs[:,0],
  'Neighbor Value': orig_green_horizontal_pairs[:,1]
})

orig_blue_horizontal_df = pd.DataFrame({
  'Pixel Value': orig_blue_horizontal_pairs[:,0],
  'Neighbor Value': orig_blue_horizontal_pairs[:,1]
})

orig_red_vertical_df = pd.DataFrame({
  'Pixel Value': orig_red_vertical_pairs[:,0],
  'Neighbor Value': orig_red_vertical_pairs[:,1]
})

orig_green_vertical_df = pd.DataFrame({
  'Pixel Value': orig_green_vertical_pairs[:,0],
  'Neighbor Value': orig_green_vertical_pairs[:,1]
})

orig_blue_vertical_df = pd.DataFrame({
  'Pixel Value': orig_blue_vertical_pairs[:,0],
  'Neighbor Value': orig_blue_vertical_pairs[:,1]
})

orig_red_diagonal_df = pd.DataFrame({
  'Pixel Value': orig_red_diagonal_pairs[:,0],
  'Neighbor Value': orig_red_diagonal_pairs[:,1]
})

orig_green_diagonal_df = pd.DataFrame({
  'Pixel Value': orig_green_diagonal_pairs[:,0],
  'Neighbor Value': orig_green_diagonal_pairs[:,1]
})

orig_blue_diagonal_df = pd.DataFrame({
  'Pixel Value': orig_blue_diagonal_pairs[:,0],
  'Neighbor Value': orig_blue_diagonal_pairs[:,1]
})

column1, column2, column3 = st.columns(3)
with column1:
  st.scatter_chart(
    orig_red_horizontal_df,
    color='#FF0000',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Red Channel Horizontal Correlation", width="content")
  st.scatter_chart(
    orig_red_vertical_df,
    color='#FF0000',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Red Channel Vertical Correlation", width="content")
  st.scatter_chart(
    orig_red_diagonal_df,
    color='#FF0000',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Red Channel Diagonal Correlation", width="content")
with column2:
  st.scatter_chart(
    orig_green_horizontal_df,
    color='#00FF00',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Green Channel Horizontal Correlation", width="content")
  st.scatter_chart(
    orig_green_vertical_df,
    color='#00FF00',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Green Channel Vertical Correlation", width="content")
  st.scatter_chart(
    orig_green_diagonal_df,
    color='#00FF00',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Green Channel Diagonal Correlation", width="content")
with column3:
  st.scatter_chart(
    orig_blue_horizontal_df,
    color='#0000FF',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Blue Channel Horizontal Correlation", width="content")
  st.scatter_chart(
    orig_blue_vertical_df,
    color='#0000FF',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Blue Channel Vertical Correlation", width="content")
  st.scatter_chart(
    orig_blue_diagonal_df,
    color='#0000FF',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Blue Channel Diagonal Correlation", width="content")
st.info("As it is evident in the scatter plots above for the original image, there is a high correllation between a pixel and its neighboring pixels in all three directions (horizontal, vertical and diagonal) and in all three channels (R, G and B).", icon="ℹ️")
st.write('_**Encrypted Image Correlation Results**_')
red_horizontal_pairs = ResultsAndDiscussions.getHorizontalPairs(st.session_state.encryptedImage[:,:,0])
green_horizontal_pairs = ResultsAndDiscussions.getHorizontalPairs(st.session_state.encryptedImage[:,:,1])
blue_horizontal_pairs = ResultsAndDiscussions.getHorizontalPairs(st.session_state.encryptedImage[:,:,2])

red_vertical_pairs = ResultsAndDiscussions.getVerticalPairs(st.session_state.encryptedImage[:,:,0])
green_vertical_pairs = ResultsAndDiscussions.getVerticalPairs(st.session_state.encryptedImage[:,:,1])
blue_vertical_pairs = ResultsAndDiscussions.getVerticalPairs(st.session_state.encryptedImage[:,:,2])

red_diagonal_pairs = ResultsAndDiscussions.getDiagonalPairs(st.session_state.encryptedImage[:,:,0])
green_diagonal_pairs = ResultsAndDiscussions.getDiagonalPairs(st.session_state.encryptedImage[:,:,1])
blue_diagonal_pairs = ResultsAndDiscussions.getDiagonalPairs(st.session_state.encryptedImage[:,:,2])

red_horizontal_df = pd.DataFrame({
  'Pixel Value': red_horizontal_pairs[:,0],
  'Neighbor Value': red_horizontal_pairs[:,1]
})
green_horizontal_df = pd.DataFrame({
  'Pixel Value': green_horizontal_pairs[:,0],
  'Neighbor Value': green_horizontal_pairs[:,1]
})
blue_horizontal_df = pd.DataFrame({
  'Pixel Value': blue_horizontal_pairs[:,0],
  'Neighbor Value': blue_horizontal_pairs[:,1]
})

red_vertical_df = pd.DataFrame({
  'Pixel Value': red_vertical_pairs[:,0],
  'Neighbor Value': red_vertical_pairs[:,1]
})
green_vertical_df = pd.DataFrame({
  'Pixel Value': green_vertical_pairs[:,0],
  'Neighbor Value': green_vertical_pairs[:,1]
})
blue_vertical_df = pd.DataFrame({
  'Pixel Value': blue_vertical_pairs[:,0],
  'Neighbor Value': blue_vertical_pairs[:,1]
})

red_diagonal_df = pd.DataFrame({
  'Pixel Value': red_diagonal_pairs[:,0],
  'Neighbor Value': red_diagonal_pairs[:,1]
})
green_diagonal_df = pd.DataFrame({
  'Pixel Value': green_diagonal_pairs[:,0],
  'Neighbor Value': green_diagonal_pairs[:,1]
})
blue_diagonal_df = pd.DataFrame({
  'Pixel Value': blue_diagonal_pairs[:,0],
  'Neighbor Value': blue_diagonal_pairs[:,1]
})

column1, column2, column3 = st.columns(3)
with column1:
  st.scatter_chart(
    red_horizontal_df,
    color='#FF0000',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Red Channel Horizontal Correlation", width="content")
  st.scatter_chart(
    red_vertical_df,
    color='#FF0000',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Red Channel Vertical Correlation", width="content")
  st.scatter_chart(
    red_diagonal_df,
    color='#FF0000',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Red Channel Diagonal Correlation", width="content")
with column2:
  st.scatter_chart(
    green_horizontal_df,
    color='#00FF00',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Green Channel Horizontal Correlation", width="content")
  st.scatter_chart(
    green_vertical_df,
    color='#00FF00',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Green Channel Vertical Correlation", width="content")
  st.scatter_chart(
    green_diagonal_df,
    color='#00FF00',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Green Channel Diagonal Correlation", width="content")
with column3:
  st.scatter_chart(
    blue_horizontal_df,
    color='#0000FF',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Blue Channel Horizontal Correlation", width="content")
  st.scatter_chart(
    blue_vertical_df,
    color='#0000FF',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Blue Channel Vertical Correlation", width="content")
  st.scatter_chart(
    blue_diagonal_df,
    color='#0000FF',
    x='Pixel Value', 
    y='Neighbor Value',
    size=1
  )
  with st.container(horizontal_alignment='center', horizontal=True):
    st.caption("Blue Channel Diagonal Correlation", width="content")
st.info("As it is evident in the scatter plots above for the encrypted image, there is a low correllation between a pixel and its neighboring pixels in all three directions (horizontal, vertical and diagonal) and in all three channels (R, G and B).", icon="ℹ️")
st.write('We find the correlation values $r_{x,y}$ with this pairs using the following formulas:')
column1, column2, column3, column4 = st.columns(4)
with column1:
  st.latex(r'''def r(x, y):
  return ResultsAndDiscussions.cov(x, y)/(np.sqrt(ResultsAndDiscussions.D(x)) * np.sqrt(ResultsAndDiscussions.D(y)))''')
with column2:
  st.latex(r'''cov(x,y) = \frac{1}{N} \sum_{i=1}^{N} (x_i - E(x))(y_i - E(y))''')
  st.code('''def cov(x, y):
  return np.mean((x - ResultsAndDiscussions.E(x)) * (y - ResultsAndDiscussions.E(y)))''')
with column3:
  st.latex(r'''D(x) = \frac{1}{N} \sum_{i=1}^{N} (x_i - E(x))^2''')
  st.code('''def D(x):
  return np.var(x)''')
with column4:
  st.latex(r'''E(x) = \frac{1}{N} \sum_{i=1}^{N} x_i''')
  st.code('''def E(x):
  return np.mean(x)''')

orig_red_horizontal_corr = ResultsAndDiscussions.r(orig_red_horizontal_pairs[:,0], orig_red_horizontal_pairs[:,1])
orig_red_vertical_corr = ResultsAndDiscussions.r(orig_red_vertical_pairs[:,0], orig_red_vertical_pairs[:,1])
orig_red_diagonal_corr = ResultsAndDiscussions.r(orig_red_diagonal_pairs[:,0], orig_red_diagonal_pairs[:,1])
orig_green_horizontal_corr = ResultsAndDiscussions.r(orig_green_horizontal_pairs[:,0], orig_green_horizontal_pairs[:,1])
orig_green_vertical_corr = ResultsAndDiscussions.r(orig_green_vertical_pairs[:,0], orig_green_vertical_pairs[:,1])
orig_green_diagonal_corr = ResultsAndDiscussions.r(orig_green_diagonal_pairs[:,0], orig_green_diagonal_pairs[:,1])
orig_blue_horizontal_corr = ResultsAndDiscussions.r(orig_blue_horizontal_pairs[:,0], orig_blue_horizontal_pairs[:,1])
orig_blue_vertical_corr = ResultsAndDiscussions.r(orig_blue_vertical_pairs[:,0], orig_blue_vertical_pairs[:,1])
orig_blue_diagonal_corr = ResultsAndDiscussions.r(orig_blue_diagonal_pairs[:,0], orig_blue_diagonal_pairs[:,1])

red_horizontal_corr = ResultsAndDiscussions.r(red_horizontal_pairs[:,0], red_horizontal_pairs[:,1])
red_vertical_corr = ResultsAndDiscussions.r(red_vertical_pairs[:,0], red_vertical_pairs[:,1])
red_diagonal_corr = ResultsAndDiscussions.r(red_diagonal_pairs[:,0], red_diagonal_pairs[:,1])
green_horizontal_corr = ResultsAndDiscussions.r(green_horizontal_pairs[:,0], green_horizontal_pairs[:,1])
green_vertical_corr = ResultsAndDiscussions.r(green_vertical_pairs[:,0], green_vertical_pairs[:,1])
green_diagonal_corr = ResultsAndDiscussions.r(green_diagonal_pairs[:,0], green_diagonal_pairs[:,1])
blue_horizontal_corr = ResultsAndDiscussions.r(blue_horizontal_pairs[:,0], blue_horizontal_pairs[:,1])
blue_vertical_corr = ResultsAndDiscussions.r(blue_vertical_pairs[:,0], blue_vertical_pairs[:,1])
blue_diagonal_corr = ResultsAndDiscussions.r(blue_diagonal_pairs[:,0], blue_diagonal_pairs[:,1])

decrypted_correlation_df = pd.DataFrame({
  'Direction': ['Horizontal', 'Vertical', 'Diagonal'],
  'Red Channel': [red_horizontal_corr, red_vertical_corr, red_diagonal_corr],
  'Green Channel': [green_horizontal_corr, green_vertical_corr, green_diagonal_corr],
  'Blue Channel': [blue_horizontal_corr, blue_vertical_corr, blue_diagonal_corr]
})

original_correlation_df = pd.DataFrame({
  'Direction': ['Horizontal', 'Vertical', 'Diagonal'],
  'Red Channel': [orig_red_horizontal_corr, orig_red_vertical_corr, orig_red_diagonal_corr],
  'Green Channel': [orig_green_horizontal_corr, orig_green_vertical_corr, orig_green_diagonal_corr],
  'Blue Channel': [orig_blue_horizontal_corr, orig_blue_vertical_corr, orig_blue_diagonal_corr]
})
column1, column2 = st.columns(2)
with column1:
  st.write("**Original Image Correlation Values**")
  st.dataframe(original_correlation_df.set_index('Direction'))
with column2:
  st.write("**Encrypted Image Correlation Values**")
  st.dataframe(decrypted_correlation_df.set_index('Direction'))
st.info("With these values, it is now evident that the encrypted image fairly removes the correlation between neighboring pixels, enhancing its security.", icon="ℹ️")
st.subheader("Information Entropy", anchor=False)
with st.container():
  st.latex('''H = - \\sum_{i=0}^{L} p(i) \\log_2 p(i)''')
  st.code('''def calculateEntropy(image):
  grayscale = ski.color.rgb2gray(image)
  histogram, _ = np.histogram(grayscale, bins=297, range=(0, 1))
  histogram = histogram / np.sum(histogram)
  entropy = 0
  for x in histogram:
    if x > 0:
      entropy += x * np.log2(x)
  return -entropy''')
  st.write(ResultsAndDiscussions.calculateEntropy(st.session_state.encryptedImage))

