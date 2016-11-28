import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('test.jpg')

ysize, xsize = image.shape[:2]
region_select = np.copy(image)
region_select2 = np.copy(image)
# color selection
thresh = 225
r_threshold = thresh
g_threshold = thresh
b_threshold = thresh

rgb_threshold = [r_threshold, g_threshold, b_threshold]

thresholds = (image[:, :, 0] < rgb_threshold[0]) | \
             (image[:, :, 1] < rgb_threshold[1]) | \
             (image[:, :, 2] < rgb_threshold[2])

# region_select[thresholds] = [0, 0, 0]

# region masking
left_bottom = [0, ysize]
right_bottom = [xsize, ysize]
apex = [xsize/2, ysize/2]

left_line = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
right_line = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
bottom_line = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))

region_threshold = (YY > (XX * left_line[0] + left_line[1])) & (YY > (XX * right_line[0] + right_line[1])) & (YY < (XX * bottom_line[0] + bottom_line[1]))

# region_select[~region_threshold] = [125, 0, 125]


# change color of lines

whites = (image[:, :, 0] > rgb_threshold[0]) | \
             (image[:, :, 1] > rgb_threshold[1]) | \
             (image[:, :, 2] > rgb_threshold[2])

region_select2[whites & region_threshold] = [255, 0, 0]

fig = plt.figure()

fig.add_subplot(2, 1, 1)
plt.imshow(region_select)
fig.add_subplot(2, 1, 2)
plt.imshow(region_select2)
plt.show()
