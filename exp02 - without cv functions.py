import cv2
import numpy as np

img = cv2.imread("input.jpg")

gray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        B, G, R = img[i, j]
        gray[i, j] = 0.114 * B + 0.587 * G + 0.299 * R


binary = np.zeros_like(gray)

for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        if gray[i, j] > 127:
            binary[i, j] = 255
        else:
            binary[i, j] = 0

cv2.imshow("Original Image", img)
cv2.imshow("Grayscale Image", gray)
cv2.imshow("Binary Image", binary)

cv2.imwrite("gray.jpg", gray)
cv2.imwrite("binary.jpg", binary)

cv2.waitKey(0)
cv2.destroyAllWindows()
 
