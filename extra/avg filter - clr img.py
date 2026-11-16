# 2 changes as in from gray 
import cv2
import numpy as np

img = cv2.imread("input.jpg")

# Create 3x3 averaging kernel
kernel = np.ones((3, 3), np.float32) / 9

h, w = img.shape[:2]

# Create output image
output = np.zeros((h, w, 3), np.uint8) #specify channel as 3

# Apply filtering manually
for i in range(1, h-1):
    for j in range(1, w-1):

        region = img[i-1:i+2, j-1:j+2]

        value = np.sum(region * kernel[:, :, np.newaxis], axis=(0, 1))   ## MAIN CHANGE HEREE

        output[i, j] = value

cv2.imshow("Original", img)
cv2.imshow("Filtered", output)

cv2.waitKey(0)
cv2.destroyAllWindows()
