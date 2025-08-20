import numpy as np

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

  return red, green, blue

def applyRowColumnTransform(red, green, blue):
  red, green, blue = RowColumnTransform1(red, green, blue)
  return red, green, blue