import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import cv2
from preprocess_images import *

# # replace black pixels with gray pixels
# def replace_black_pixels_with_gray_pixels(image):
#     image = np.array(image)
#     image[image == 0] = 128
#     return Image.fromarray(image)

# img = Image.open('10_left.png')
# img = replace_black_pixels_with_gray_pixels(img)
# img.show()
image = cv2.imread('10_left.png')
# cv2.imshow('image', image)
# cv2.waitKey(0)
trim_img = trim(image)
cv2.imshow('trim_img', trim_img)
cv2.waitKey(0)

