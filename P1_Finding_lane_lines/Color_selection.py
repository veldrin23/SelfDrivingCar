import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


image = mpimg.imread('test.jpg')
print('This image is:',  type(image),
      'with dimensions:', image.shape)

