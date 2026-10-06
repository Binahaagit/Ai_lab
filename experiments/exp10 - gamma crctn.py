import cv2
import numpy as np

img = cv2.imread("input.jpg")

gamma = 0.5   #change gamma value here

#for colour image
normalized = img / 255.0
gamma_clr = np.power(normalized, gamma)
gamma_clr = np.uint8(gamma_clr * 255)

#for gray img
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

normalized_gray = gray / 255.0
gamma_gray = np.power(normalized_gray, gamma)
gamma_gray = np.uint8(gamma_gray * 255)

# Display
cv2.imshow("Original Colour", img)
cv2.imshow("Gamma Colour", gamma_clr)

cv2.imshow("Original Grayscale", gray)
cv2.imshow("Gamma Grayscale", gamma_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Gamma correction → changes image brightness/intensity
# Formula: s = r^gamma
# Normalize → convert pixel values from 0–255 to 0–1
# np.power() → applies gamma power to each pixel
# Multiply by 255 → convert values back to 0–255
# np.uint8 → converts result to 8-bit image
# gamma < 1 → image becomes brighter
# gamma = 1 → image remains unchanged
# gamma > 1 → image becomes darker
# Same process works for grayscale and colour images
# Colour image → gamma correction is applied separately to B, G and R channels
