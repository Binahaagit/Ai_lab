import cv2
import numpy as np

img1 = cv2.imread("input.jpg")
img2 = cv2.imread("input2.jpg")

# Resize second image to match first image
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Bitwise AND
bitwise_and = cv2.bitwise_and(img1, img2)

# Bitwise OR
bitwise_or = cv2.bitwise_or(img1, img2)

# Bitwise XOR
bitwise_xor = cv2.bitwise_xor(img1, img2)

# Bitwise NOT
bitwise_not = cv2.bitwise_not(img1)

# Create a mask
mask = np.zeros(img1.shape[:2], np.uint8)
mask[50:200, 50:200] = 255

# Apply mask using AND
masked_image = cv2.bitwise_and(img1, img1, mask=mask)

# Display results
cv2.imshow("Original", img1)
cv2.imshow("AND", bitwise_and)
cv2.imshow("OR", bitwise_or)
cv2.imshow("XOR", bitwise_xor)
cv2.imshow("NOT", bitwise_not)
cv2.imshow("Mask", mask)
cv2.imshow("Masked Image", masked_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
