import cv2
import numpy as np

# Create two 300x300 black images
img1 = np.zeros((300, 300), dtype=np.uint8)
img2 = np.zeros((300, 300), dtype=np.uint8)

cv2.rectangle(img1, (25, 25), (275, 275), 255, -1)  # Fill rectangle from (25,25) to (275,275) with white (-1 means FILL)
cv2.circle(img2, (150, 150), 150, 255, -1)   # Fill circle with center (150,150), radius 150 with white(255)

# Logical operations
AND = cv2.bitwise_and(img1, img2)
OR = cv2.bitwise_or(img1, img2)
XOR = cv2.bitwise_xor(img1, img2)

# Display
cv2.imshow("Rectangle", img1)
cv2.imshow("Circle", img2)
cv2.imshow("AND", AND)
cv2.imshow("OR", OR)
cv2.imshow("XOR", XOR)

cv2.waitKey(0)
cv2.destroyAllWindows()

#if we can use rectangle n circle images directly to perform operns, then no need to create, just use those images using imread
