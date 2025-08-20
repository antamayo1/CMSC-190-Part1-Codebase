import numpy as np

def get_RGB_channels(image):
  red_channel = image[:, :, 0]
  green_channel = image[:, :, 1]
  blue_channel = image[:, :, 2]
  return red_channel, green_channel, blue_channel

def get_inverted_RGB_channels(image):
  red_channel = 255 - image[:, :, 0]
  green_channel = 255 - image[:, :, 1]
  blue_channel = 255 - image[:, :, 2]
  return red_channel, green_channel, blue_channel

def get_RGB_channels_as_images(red, green, blue, image):
  red_image = np.zeros_like(image)
  green_image = np.zeros_like(image)
  blue_image = np.zeros_like(image)
  red_image[:, :, 0] = red
  green_image[:, :, 1] = green
  blue_image[:, :, 2] = blue
  return red_image, green_image, blue_image