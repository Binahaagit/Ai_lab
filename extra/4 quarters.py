import cv2
import numpy as np

img = cv2.imread('image.jpg')

# Gaussian
g = cv2.GaussianBlur(img, (5,5), 0)

# Laplacian
l = cv2.Laplacian(img, cv2.CV_64F)
l = cv2.convertScaleAbs(l)

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Negative
neg = 255 - img

# 2x2 Grid
grid = np.vstack((
    np.hstack((g, l)),
    np.hstack((cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), neg))
))

cv2.imshow('Output', grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
