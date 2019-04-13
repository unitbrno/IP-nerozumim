import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from path_provider import PathProvider

paths = PathProvider().get_paths_dict()

img  = cv.imread(paths[6][1], cv.IMREAD_ANYDEPTH)

img_scaled = cv.normalize(img, dst=None, alpha=0, beta=65535, norm_type=cv.NORM_MINMAX)

plt.subplot(311).set_title('Original histogram')

plt.hist(img_scaled.ravel())

plt.subplot(312).set_title('Normalized histogram')

plt.hist(img.ravel())


plt.subplot(313).set_title('Histogram after conversion to 8bit')

img8 = (img_scaled/256).astype('uint8')

plt.hist(img8.ravel())

plt.show()

