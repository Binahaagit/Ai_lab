# Remember - RGB versus BGR, absolute difference, and np.all()

import cv2
import numpy as np

img = cv2.imread("input.jpg")

# Read original and replacement RGB colours
r, g, b = map(int, input("Enter original RGB: ").split())
r1, g1, b1 = map(int, input("Enter replacement RGB: ").split())

# Convert RGB to BGR (OpenCV format)
old = np.array([b, g, r])
new = np.array([b1, g1, r1])

# Replace colour with tolerance 10
for i in range(img.shape[0]):
    for j in range(img.shape[1]):

        pixel = img[i, j]

        if np.all(np.abs(pixel.astype(int) - old) <= 10):
            img[i, j] = new

cv2.imshow("Modified Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
