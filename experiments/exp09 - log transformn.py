import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

c = 255 / np.log(1 + 255)                      
log_transformed = c * np.log(1 + img)         # Log transformation: s = c * log(1 + r)

log_transformed = np.uint8(log_transformed)

cv2.imshow("Original", img)
cv2.imshow("Log Transform", log_transformed)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Log transformation → enhances details in dark regions
# Formula: s = c * log(1 + r)
# c = 255 / log(256) → scaling constant
# +1 → avoids log(0)
# np.log() → calculates logarithm
# uint8 → converts result to 0–255 image format




