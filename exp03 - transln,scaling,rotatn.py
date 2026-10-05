import cv2
import numpy as np

img = cv2.imread("input.jpg")
h, w = img.shape[:2]

# Translation
M = np.float32([[1, 0, 100], [0, 1, 50]])
translated = cv2.warpAffine(img, M, (w, h))

# Scaling
scaled = cv2.resize(img, None, fx=1.5, fy=1.5)

# Rotation
M = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1)
rotated = cv2.warpAffine(img, M, (w, h))

cv2.imshow("Original", img)
cv2.imshow("Translated", translated)
cv2.imshow("Scaled", scaled)
cv2.imshow("Rotated", rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()


