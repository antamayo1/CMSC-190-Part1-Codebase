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
  def reverseStepOne(red, green, blue):
    size = red.shape[0]

    for col in range(1, size):
      if col%2 == 1:
        green[:, col] = 255 - green[:, col]

    for row in range(1, size):
      if row%2 == 1:
        red[row, :] = 255 - red[row, :]

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
    for col in range(size-1, 0, -1):
      if (col-1)%5 == 0:
        tmp = green[:, col].copy()
        green[:, col] = blue[:, col]
        blue[:, col] = tmp

    for col in range(size-1, 0, -1):
      if (col-1)%3 == 0:
        tmp = red[:, col].copy()
        red[:, col] = blue[:, col]
        blue[:, col] = tmp

    for col in range(size-1, 0, -1):
      tmp = red[:, col].copy()
      red[:, col] = green[:, col]
      green[:, col] = tmp

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

  @staticmethod
  def reverseStepFour(red, green, blue):
    size = red.shape[0]

    for row in range(size):
      red[row, :] = np.roll(red[row, :], -row)

    for row in range(0, size, 3):
      blue[row, :] = np.roll(blue[row, :], -row)

    for row in range(0, size, 5):
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

  def getChaoticImage(map_type, length, size):
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
      new_pixel[idx] = str(1 - int(pixel[idx]))
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
  def E(x, y, image):
    
    horizontal = 0
    horizontal_total = 0
    vertical = 0
    vertical_total = 0
    diagonal = 0
    diagonal_total = 0

    # horizontal check
    try:
      horizontal_total += image[x - 1, y]
      horizontal += 1
    except:
      pass
    try:
      horizontal_total += image[x + 1, y]
      horizontal += 1
    except:
      pass

    # vertical check
    try:
      vertical_total += image[x, y - 1]
      vertical += 1
    except:
      pass
    try:
      vertical_total += image[x, y + 1]
      vertical += 1
    except:
      pass

    # diagonal check
    try:
      diagonal_total += image[x - 1, y - 1]
      diagonal += 1
    except:
      pass
    try:
      diagonal_total += image[x - 1, y + 1]
      diagonal += 1
    except:
      pass
    try:
      diagonal_total += image[x + 1, y - 1]
      diagonal += 1
    except:
      pass
    try:
      diagonal_total += image[x + 1, y + 1]
      diagonal += 1
    except:
      pass

    return horizontal_total/horizontal, vertical_total/vertical, diagonal_total/diagonal
