import cv2
import matplotlib.pyplot as plt
# Read image as grayscale
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# Histogram equalization
equalized = cv2.equalizeHist(img)

# Display images
plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.hist(img.ravel(), 256, [0, 256])
plt.title("Original Histogram")

plt.subplot(2, 2, 3)
plt.imshow(equalized, cmap="gray")
plt.title("Equalized Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.hist(equalized.ravel(), 256, [0, 256])
plt.title("Equalized Histogram")

plt.tight_layout()
plt.show()

# subplot(2,2,1) → 2 rows, 2 columns, position 1
# subplot(2,2,2) → position 2
# ravel() → converts 2D image pixels into 1D array
# 256 → 256 possible grayscale intensity values (0–255)
# [0,256] → histogram intensity range
# cmap="gray" → Display image in grayscale
# axis("off") → Hide x and y axes
