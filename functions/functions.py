def get_RGB_channels(image):
  red_channel = image[:, :, 0]
  green_channel = image[:, :, 1]
  blue_channel = image[:, :, 2]
  return red_channel, green_channel, blue_channel