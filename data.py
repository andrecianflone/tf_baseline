
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../data/'))

# from mnist_data import get_data

# def get_mnist():
  # return get_data()

def disp_img(matrix):
  """ Display a numpy array as image"""
  assert len(matrix.shape) == 2
  from matplotlib import pyplot as plt
  plt.imshow(matrix, cmap=plt.get_cmap('gray'))
  plt.show()

