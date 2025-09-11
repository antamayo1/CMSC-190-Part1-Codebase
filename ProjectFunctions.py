import numpy as np
import skimage as ski

class Tools:
  @staticmethod
  def channelsAsImages(red, green, blue, image):
    red_image = np.zeros_like(image)
    green_image = np.zeros_like(image)
    blue_image = np.zeros_like(image)
    red_image[:, :, 0] = red
    green_image[:, :, 1] = green
    blue_image[:, :, 2] = blue
    return red_image, green_image, blue_image
  
  @staticmethod
  def mergeChannels(red, green, blue):
    return np.stack((red, green, blue), axis=-1)

class ColorSplit:

  @staticmethod
  def getChannels(image):
    redChannel = image[:, :, 0]
    greenChannel = image[:, :, 1]
    blueChannel = image[:, :, 2]
    return redChannel, greenChannel, blueChannel

  @staticmethod
  def getInvertedChannels(image):
    redChannel = 255 - image[:, :, 0]
    greenChannel = 255 - image[:, :, 1]
    blueChannel = 255 - image[:, :, 2]
    return redChannel, greenChannel, blueChannel

class RowColumnTransform:

  @staticmethod
  def stepOne(red, green, blue):
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

    return red, green, blue
  
  @staticmethod
  def reverseStepOne(red, green, blue):
    size = red.shape[0]

    for col in range(1, size):
      if col%2 == 1:
        green[:, col] = 255 - green[:, col]

    for row in range(size):
      if row%2 == 1:
        red[1:row, :] = 255 - red[1:row, :] 

    for col in range(1, size):
      if (col-1)%3 == 0:
        blue[:, col] = 255 - blue[:, col]

    return red, green, blue

  @staticmethod
  def stepTwo(red, green, blue):
    size = red.shape[0]

    # The R and G are switched in every column
    for col in range(1, size):
      tmp = red[:, col].copy()
      red[:, col] = green[:, col]
      green[:, col] = tmp

    # The R and B channels are switched in every third column
    for col in range(1, size):
      if (col-1)%3 == 0:
        tmp = red[:, col].copy()
        red[:, col] = blue[:, col]
        blue[:, col] = tmp

    # The G and B channels are switched in every fifth column
    for col in range(1, size):
      if (col-1)%5 == 0:
        tmp = green[:, col].copy()
        green[:, col] = blue[:, col]
        blue[:, col] = tmp

    return red, green, blue

  @staticmethod
  def reverseStepTwo(red, green, blue):
    size = red.shape[0]
    # Reverse order for proper undoing
    for col in range(size-1, 1, -1):
      if (col-1)%5 == 0:
        tmp = green[:, col].copy()
        green[:, col] = blue[:, col]
        blue[:, col] = tmp

    for col in range(size-1, 1, -1):
      if (col-1)%3 == 0:
        tmp = red[:, col].copy()
        red[:, col] = blue[:, col]
        blue[:, col] = tmp

    for col in range(size-1, 1, -1):
      tmp = red[:, col].copy()
      red[:, col] = green[:, col]
      green[:, col] = tmp

    return red, green, blue

  @staticmethod
  def stepThree(red, green, blue):
    # Inversion operation for each channel
    red[1:] = 255 - red[1:]
    green[1:] = 255 - green[1:]
    blue[1:] = 255 - blue[1:]
    return red, green, blue
  
  @staticmethod
  def stepFour(red, green, blue):
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

    return red, green, blue

  @staticmethod
  def reverseStepFour(red, green, blue):
    size = red.shape[0]

    for row in range(1, size):
      red[row, :] = np.roll(red[row, :], -row)

    for row in range(1, size, 3):
      blue[row, :] = np.roll(blue[row, :], -row)

    for row in range(1, size, 5):
      green[row, :] = np.roll(green[row, :], -row)

    return red, green, blue

class ChaoticSequence:

  def XOR_images(imageA, imageB):
    return np.bitwise_xor(imageA, imageB)

  def getNext_LCM(lastValue):
    r = 4
    return r * lastValue * (1 - lastValue)

  def getNext_SCM(lastValue):
    r = 4
    return r/4*np.sin(np.pi * lastValue)
  
  def get_Next_LSCM(lastValue):
    r = 4
    b = 4
    return np.sin(r*np.pi*(1-lastValue)*b*lastValue)

  def getSequenceAsImage(sequence, size):
    chaoticImage = np.zeros((size, size, 3))
    idx = 0
    for i in range(3):
      for row in range(0, size):
        for col in range(0, size):
          chaoticImage[row, col, i] = sequence[idx]
          idx += 1
    chaoticImage = (chaoticImage*255).astype(np.uint8)
    return chaoticImage

  def getSequenceOnly(map_type, length):
    sequence = []
    initial_r_value = 0.4
    for _ in range(length):
      if map_type == "LCM":
        initial_r_value = ChaoticSequence.getNext_LCM(initial_r_value)
      elif map_type == "SCM":
        initial_r_value = ChaoticSequence.getNext_SCM(initial_r_value)
      else:
        initial_r_value = ChaoticSequence.get_Next_LSCM(initial_r_value)
      sequence.append(initial_r_value)
    return sequence

  def getChaoticImage(map_type, length, size):
    sequence = []
    initial_r_value = 0.63
    for _ in range(length):
      if map_type == "LCM":
        initial_r_value = ChaoticSequence.getNext_LCM(initial_r_value)
      elif map_type == "SCM":
        initial_r_value = ChaoticSequence.getNext_SCM(initial_r_value)
      else:
        initial_r_value = ChaoticSequence.get_Next_LSCM(initial_r_value)
      sequence.append(initial_r_value)
    return ChaoticSequence.getSequenceAsImage(sequence, size)

class CellularAutomata:
  def applyXOR(pixel):
    new_pixel = pixel
    new_pixel = list(new_pixel)
    for idx in range(4):
      if pixel[idx] == pixel[(idx+1)]:
        new_pixel[idx] = '0'
      else:
        new_pixel[idx] = '1'
    return ''.join(new_pixel)

  def applyinvert(pixel):
    new_pixel = pixel
    new_pixel = list(new_pixel)
    for idx in range(4, 8):
      if pixel[idx] == '0':
        new_pixel[idx] = '1'
      else:
        new_pixel[idx] = '0'
    return ''.join(new_pixel)

  def applyCATransform(pixel):
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
    return image

class ResultsAndDiscussions:
  def calculateEntropy(image):
    grayscale = ski.color.rgb2gray(image)
    histogram, _ = np.histogram(grayscale, bins=315, range=(0, 1))
    histogram = histogram / np.sum(histogram)
    entropy = 0
    for x in histogram:
      if x > 0:
        entropy += x * np.log2(x)
    return -entropy
  
  def getHorizontalPairs(image):
    pairs = []
    for row in range(image.shape[0]):
      for col in range(image.shape[1]-1):
        pairs.append([image[row, col], image[row, col+1]])
    return np.array(pairs)

  def getVerticalPairs(image):
    pairs = []
    for col in range(image.shape[1]):
      for row in range(image.shape[0]-1):
        pairs.append([image[row, col], image[row+1, col]])
    return np.array(pairs)
  
  def getDiagonalPairs(image):
    pairs = []
    for row in range(image.shape[0]-1):
      for col in range(image.shape[1]-1):
        pairs.append([image[row, col], image[row+1, col+1]])
    return np.array(pairs)
  
  def E(x):
    return np.mean(x)
  
  def D(x):
    return np.var(x)
  
  def cov(x, y):
    x_mean = ResultsAndDiscussions.E(x)
    y_mean = ResultsAndDiscussions.E(y)
    return np.mean((x - x_mean) * (y - y_mean))

  def r(x, y):
    return ResultsAndDiscussions.cov(x, y)/(np.sqrt(ResultsAndDiscussions.D(x)) * np.sqrt(ResultsAndDiscussions.D(y)))