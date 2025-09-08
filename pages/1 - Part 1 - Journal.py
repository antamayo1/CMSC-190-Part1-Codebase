import streamlit as st

st.set_page_config(page_title="CMSC 190", layout="wide")
with st.sidebar:
  st.title("`CMSC 190 Notebook`")
  st.write("A.J.N.T")

with st.container(border=True):
  st.header('Part 1 Journal', anchor=False)

  st.subheader('August 19 2025, Tuesday', anchor=False)
  with st.expander("View Details"):
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
    return red_channel, green_channel, blue_channel''', language='python')
    with code_2:
      st.write("**Inverted Implementation**")
      st.code('''
  def get_RGB_channels(image):
    red_channel = 255 - image[:, :, 0]
    green_channel = 255 - image[:, :, 1]
    blue_channel = 255 - image[:, :, 2]
    return red_channel, green_channel, blue_channel''', language='python')
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

  st.subheader('August 20 2025, Wednesday', anchor=False)
  with st.expander("View Details"):
    st.markdown('''
    Continued reading the preliminaries of the reference paper and added notes about the `chaotic maps` like `logistic maps` and `sine chaotic maps`.
    Will continue reading about the reference paper's introduced chaotic map, `Logistic-Sine Chaotic Map`.
                
    I have also started with the next step in the methodology which is the `Row and Column Transformation` which has confusing terms.
    
    **GDOCS UPDATE**:
    > None. Will try adding some details there soon.
    ''')

  st.subheader('September 8 2025, Monday', anchor=False)
  with st.expander("View Details"):
    st.markdown('''
    Long overdue update. Finished implementing the encyption algorihm (from my understanding of the paper) and have started reviewing the results and discussions section of the paper.
    The histogram test does not seem to match the expected result but at least the original image and decrypted image have the same histogram. Reviewing the next test in the paper which is the correlation test with the following equations:

    * $$r_{x,y}=\\frac{cov(x,y)}{\\sqrt{D(x)}\\sqrt{D(y)}}$$
    * $$cot(x,y)=\\frac{1}{N}\sum_{i=1}(x_i-E(x))(y_i-E(y))$$
    * $$D(x)=\\frac{1}{N}\\sum_{i=1}^N (x_i-E(x))^2$$
    * $$E(x)=\\frac{1}{N}\\sum_{i=1}^N x_i$$

    I don't even know if $$cov(x,y)$$ is the same as $$cot(x,y)$$ in the paper. For what I remember from the equations, $$E(x)$$ is the mean, $$D(x)$$ is the variance. How do I implement all of these in the context of an image where the paper mentions about neighboring pixels?
                
    **GDOCS UPDATE**:
    > Finally added the encryption and decryption results.
    > I have also added the link to the notebook used for the implementation.
    ''')

with st.container(border=True):
  st.header('Notes')

  st.subheader('Requirements.txt', anchor=False)
  with st.expander("View Requirements"):
    st.code('''
  python==3.13.7
  numpy==2.3.2
  scikit-image==0.25.2''')
    
  st.subheader('Channel Split', anchor=False)
  with st.expander("View Notes"):
    st.write('''
    * A colored image is an $M \\times N \\times 3$ array, where $M$ is the height, $N$ is the width, and $3$ represents the RGB channels.
    * The channel split process separates the image into its individual R, G, and B components.
    * The current implementation uses `scikit-image` to read the image into a numpy array.
    > This definition still does not match the expected output in the reference paper.
    ''')

  st.subheader('Chaotic Maps', help='Use in image encryption', anchor=False)
  with st.expander("View Notes"):
    st.write('''
    * **Definition**: A chaotic map is a _cryptosystem_ that utilizes _nonlinear functions_ to create complex random sequences for
      encrypting data. The output of chaotic maps is heavily influenced by control parameters and initial settings, making them
      effective tools for cryptography.
    > Although chaotic maps create complex random sequences, if the control parameter and the initial settings are known, then
      one can trace back the sequence to its origin.
    * **Logistic mapping** is a common nonlinear dynamic system model, the mathematical expression of one-dimensional logistic mapping is

      $x_{n+1} = r \\times x_n \\times (1 - x_n)$

      where $r$ is the **control parameter**, $r\\in(0, 4]$ the sequence is in chaotic state only when $3.569945627 < r \\leq 4$, the sequence
      $x_n$ is in chaotic state.
    > As $r$ starts from and increases, it oscillates from 2 values, then 4 values, then 8 values, and so on, following a
      pattern of doubling for which they call **Period Doubling Bifurcation**. When $r$ reaches more than $3.569945627$, the sequence
      becomes chaotic as the bifurcation is now unpredictable with an unknown $r$. That is why, $r$ reaching the range $(3.569945627, 4]$
      considers the system to be in a chaotic state.
    * **Sine Chaotic Map** is a nonlinear dynamical system model based on sinusoidal functions, and the expression of Sine Chaotic map is

      $x_{i+1} = \\frac{a}{4} \\times \\sin(\\pi \\times x_i)$

      where $a$ is the **control parameter** which is typically 4.
    ''')