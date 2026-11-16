# ONLY CHANGE IN KERNAL
import cv2
import numpy as np

img = cv2.imread("input.jpg")

kernel = np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
], np.float32) / 256

h, w = img.shape[:2]
output = np.zeros((h, w, 3), np.uint8)

for i in range(2, h-2):
    for j in range(2, w-2):

        region = img[i-2:i+3, j-2:j+3]

        value = np.sum(
            region * kernel[:, :, np.newaxis],
            axis=(0, 1)
        )

        output[i, j] = value

cv2.imshow("Original", img)
cv2.imshow("Filtered", output)

cv2.waitKey(0)
cv2.destroyAllWindows()
