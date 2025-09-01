import numpy as np

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
    for row in range(1, size):
      if row%2 == 1:
        red[row, :] = 255 - red[row, :]

    # every third column in the B channel is inverted
    for col in range(1, size):
      if (col-1)%3 == 0:
        blue[:, col] = 255 - blue[:, col]

    return red, green, blue
  
  @staticmethod
  def stepTwo(red, green, blue):
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

    return red, green, blue

  @staticmethod
  def stepThree(red, green, blue):
    # Inversion operation for each channel
    red = 255 - red
    green = 255 - green
    blue = 255 - blue
    return red, green, blue
  
  @staticmethod
  def stepFour(red, green, blue):
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

    return red, green, blue

class ChaoticSequence:

  def getNext(lastValue):
    r = 4
    return r * lastValue * (1 - lastValue)

  def getChaoticImage(size):
    chaoticImage = np.zeros((size, size, 3))
    chaoticImage[0, 0, 0] = 4
    for i in range(3):
      for row in range(0, size):
        for col in range(0, size):
          chaoticImage[row, col, i] = ChaoticSequence.getNext(chaoticImage[row, col, i])
    return chaoticImage

  def getSequence(length):
    sequence = []
    lastValue = 0.4
    for _ in range(length):
      lastValue = ChaoticSequence.getNext(lastValue)
      sequence.append(lastValue)
    return sequence