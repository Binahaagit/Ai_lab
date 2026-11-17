import cv2
import numpy as np

# Step 1: Read two images in grayscale
img1 = cv2.imread("image1.jpg", 0)
img2 = cv2.imread("image2.jpg", 0)

# Step 2: Resize image2 to match image1 dimensions
h, w = img1.shape
img2 = cv2.resize(img2, (w, h))

# Step 3: Create empty output images
AND = np.zeros((h, w), np.uint8)
OR = np.zeros((h, w), np.uint8)
XOR = np.zeros((h, w), np.uint8)

# Step 4: Perform bitwise operations manually
for i in range(h):
    for j in range(w):

        # Bitwise AND operation
        AND[i, j] = img1[i, j] & img2[i, j]

        # Bitwise OR operation
        OR[i, j] = img1[i, j] | img2[i, j]

        # Bitwise XOR operation
        XOR[i, j] = img1[i, j] ^ img2[i, j]

# Step 5: Display the output images
cv2.imshow("Image 1", img1)
cv2.imshow("Image 2", img2)
cv2.imshow("AND Operation", AND)
cv2.imshow("OR Operation", OR)
cv2.imshow("XOR Operation", XOR)

# Step 6: Wait and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
