import cv2
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg")

# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Split HSV channels
h, s, v = cv2.split(hsv)

# Equalize V channel
v_equalized = cv2.equalizeHist(v)

# Merge channels back
hsv_equalized = cv2.merge([h, s, v_equalized])

# Convert HSV back to BGR
equalized = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)

# Display original image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# Original V histogram
plt.subplot(2, 2, 2)
plt.hist(v.ravel(), 256, [0, 256])
plt.title("Original Histogram")

# Display equalized image
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB))
plt.title("Equalized Image")
plt.axis("off")

# Equalized V histogram
plt.subplot(2, 2, 4)
plt.hist(v_equalized.ravel(), 256, [0, 256])
plt.title("Equalized Histogram")

plt.tight_layout()
plt.show()

# BGR → HSV → separate H, S, V
# H = Hue → color
# S = Saturation → intensity of color
# V = Value → brightness

# Equalize only V so that the colors are not distorted

# Equalize V → merge H, S, V → convert HSV back to BGR
