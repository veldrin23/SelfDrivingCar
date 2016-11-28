import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np



image = mpimg.imread('test.jpg')
print('This image is:',  type(image),
      'with dimensions:', image.shape)

xsize = image.shape[1]
ysize = image.shape[0]
color_select = np.copy(image)


t_value = 200
red_threshold = t_value
green_threshold = t_value
blue_threshold = t_value
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

thresholds = (image[:, :, 0] < rgb_threshold[0]) | \
              (image[:, :, 1] < rgb_threshold[1]) | \
              (image[:, :, 2] < rgb_threshold[2])


color_select[thresholds] = [0, 0, 0]  # makes it black
# print(sum(sum(thresholds)))  # 0

fig = plt.figure()

fig.add_subplot(1, 2, 1)
plt.imshow(image)

fig.add_subplot(1, 2, 2)
plt.imshow(color_select)

plt.show()  # should look the same
