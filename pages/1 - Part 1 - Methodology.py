import streamlit as st
import skimage as ski
import ChannelSplitFunc
import RowColumnTransformFunc

st.set_page_config(page_title="CMSC 190", layout="wide")
with st.sidebar:
  st.title("`CMSC 190 Notebook`")
  st.write("A.J.N.T")

research_paper = "A Color Image Encryption Scheme Utilizing a Logistic-Sine Chaotic Map and Cellular Automata"

st.title(research_paper)
st.write('By: Shiji Sun, Wenzhong Yang, Yabo Yin, Xiaodan Tian, Guanghan Li, Xiangxin Deng')
st.markdown("---")

image_path = "static/Jelly.tiff"
image = ski.io.imread(image_path)

st.subheader("Channel Split")
with st.expander("View Details"):
  col1, col2 = st.columns(2)
  with col1:
    st.info("This is using the normal channel split method.")
    st.code('''
def get_RGB_channels(image):
  red_channel = image[:, :, 0]
  green_channel = image[:, :, 1]
  blue_channel = image[:, :, 2]
  return red_channel, green_channel, blue_channel''', language="python")
    red_image, green_image, blue_image = ChannelSplitFunc.get_RGB_channels(image)
    original_column, red_column, green_column, blue_column = st.columns(4)
    with original_column:
      st.image(image, caption="Original Image", use_container_width=True)
    with red_column:
      st.image(red_image, caption="Red Channel", use_container_width=True)
    with green_column:
      st.image(green_image, caption="Green Channel", use_container_width=True)
    with blue_column:
      st.image(blue_image, caption="Blue Channel", use_container_width=True)
    red_colored, green_colored, blue_colored = ChannelSplitFunc.get_RGB_channels_as_images(red_image, green_image, blue_image, image)
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
def get_inverted_RGB_channels(image):
  red_channel = 255 - image[:, :, 0]
  green_channel = 255 - image[:, :, 1]
  blue_channel = 255 - image[:, :, 2]
  return red_channel, green_channel, blue_channel''', language="python")
    red_image, green_image, blue_image = ChannelSplitFunc.get_inverted_RGB_channels(image)
    original_column, red_column, green_column, blue_column = st.columns(4)
    with original_column:
      st.image(image, caption="Original Image", use_container_width=True)
    with red_column:
      st.image(red_image, caption="Red Channel", use_container_width=True)
    with green_column:
      st.image(green_image, caption="Green Channel", use_container_width=True)
    with blue_column:
      st.image(blue_image, caption="Blue Channel", use_container_width=True)
    red_colored, green_colored, blue_colored = ChannelSplitFunc.get_RGB_channels_as_images(red_image, green_image, blue_image, image)
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
  st.code('''
def RowColumnTransform1(red, green, blue):
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
  red, green, blue = ChannelSplitFunc.get_RGB_channels(image)
  red, green, blue = RowColumnTransformFunc.RowColumnTransform1(red, green, blue)
  red_image, green_image, blue_image = ChannelSplitFunc.get_RGB_channels_as_images(red, green, blue, image)
  col1, col2, col3 = st.columns(3)
  with col1:
    st.image(red_image, caption="Red Channel (after step 1)", use_container_width=True)
  with col2:
    st.image(green_image, caption="Green Channel (after step 1)", use_container_width=True)
  with col3:
    st.image(blue_image, caption="Blue Channel (after step 1)", use_container_width=True)
  st.error('''Really not sure with this step''')
  st.write('''
Step 2: The R and G channels are switched in every column. The R and B channels are switched in every third column. G
and B channels switched in every 5 column.''')

st.subheader("Generating chaotic sequences")
with st.expander("View Details"):
  st.write("Soon")

st.subheader("Cellular automata process")
with st.expander("View Details"):
  st.write("Soon")

st.subheader("Decryption process")
with st.expander("View Details"):
  st.write("Soon")
