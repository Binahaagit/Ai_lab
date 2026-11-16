# ONLY CHANGE - using convertScaleAbs at the end and the kernel
import cv2
import numpy as np

img = cv2.imread("input.jpg", 0)

# Laplacian kernel
kernel = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
], np.float32)

h, w = img.shape
output = np.zeros((h, w), np.float32)

for i in range(1, h-1):
    for j in range(1, w-1):

        region = img[i-1:i+2, j-1:j+2]

        value = np.sum(region * kernel)

        output[i, j] = value

# Convert negative and positive values to displayable format
output = cv2.convertScaleAbs(output)

cv2.imshow("Original", img)
cv2.imshow("Laplacian", output)

cv2.waitKey(0)
cv2.destroyAllWindows()
