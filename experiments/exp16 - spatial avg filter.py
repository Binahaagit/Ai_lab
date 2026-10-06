import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg")

# Create averaging filters
k3 = np.ones((3, 3), np.float32) / 9
k5 = np.ones((5, 5), np.float32) / 25
k7 = np.ones((7, 7), np.float32) / 49

# Apply spatial averaging filters
filter3 = cv2.filter2D(img, -1, k3)
filter5 = cv2.filter2D(img, -1, k5)
filter7 = cv2.filter2D(img, -1, k7)

# Display original and filtered images
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))   #remember to change to rgb
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(filter3, cv2.COLOR_BGR2RGB))
plt.title("3x3 Filter")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(filter5, cv2.COLOR_BGR2RGB))
plt.title("5x5 Filter")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(filter7, cv2.COLOR_BGR2RGB))
plt.title("7x7 Filter")
plt.axis("off")

plt.tight_layout()  # Prevents titles and plots from overlapping
plt.show()

# Spatial averaging filter → smooths/blurs the image
# 3x3 → average of 9 pixels
# 5x5 → average of 25 pixels
# 7x7 → average of 49 pixels
# Larger kernel → more smoothing
# cv2.filter2D() → applies the filter
# np.ones() → creates a matrix of ones
# Divide by number of elements → gives equal averaging weights
