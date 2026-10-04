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
 


# img.shape[0] → Height (no of rows)
# img.shape[1] → Width (no of columns)
# np.zeros() → Creates an array filled with 0 (black)
# dtype = data type
# uint8 → Pixel values range from 0 to 255
# np.zeros_like(gray) → Creates a zero-filled array like gray
# >127 → White (255)
# ≤127 → Black (0)
