# ONLY CHANGE IN KERNEL
import cv2
import numpy as np

img = cv2.imread("input.jpg", 0)

# Gaussian kernel
kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], np.float32) / 16

h, w = img.shape
output = np.zeros((h, w), np.uint8)

for i in range(1, h-1):
    for j in range(1, w-1):

        region = img[i-1:i+2, j-1:j+2]

        value = np.sum(region * kernel)

        output[i, j] = value

cv2.imshow("Original", img)
cv2.imshow("Gaussian Filter", output)

cv2.waitKey(0)
cv2.destroyAllWindows()
