
import cv2
import numpy as np

img = cv2.imread("input.jpg")

# Flip images
hflip = img[:, ::-1]
vflip = img[::-1, :]

# Create white margins
margin = np.ones((img.shape[0], 10, 3), np.uint8) * 255

# Create collage
collage = np.hstack((margin, img, margin, hflip, margin, vflip, margin))

cv2.imshow("Original", img)
cv2.imshow("Horizontal Flip", hflip)
cv2.imshow("Vertical Flip", vflip)
cv2.imshow("Collage", collage)

cv2.waitKey(0)
cv2.destroyAllWindows()
