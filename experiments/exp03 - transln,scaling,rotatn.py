import cv2
import numpy as np

img = cv2.imread("input.jpg")
h, w = img.shape[:2]

# Translation
M = np.float32([[1, 0, 100], [0, 1, 50]])        #change 100 and 50 to change transln value of x and y axis respectively
translated = cv2.warpAffine(img, M, (w, h))

# Scaling
scaled = cv2.resize(img, None, fx=1.5, fy=1.5)     #change fx,fy values to control scaling

# Rotation
M = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1)   #alter the value of 45 alone and give any angle of rotatn of ur choice
rotated = cv2.warpAffine(img, M, (w, h))

cv2.imshow("Original", img)
cv2.imshow("Translated", translated)
cv2.imshow("Scaled", scaled)
cv2.imshow("Rotated", rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()


# cv2.warpAffine() → Applies translation(moves image)
# cv2.resize() → Scales/resizes image
# cv2.getRotationMatrix2D() → Creates rotation matrix
# img.shape[:2] → takes height and width of img, ignores colour channels
