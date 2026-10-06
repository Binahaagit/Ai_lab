import cv2

img = cv2.imread("input.jpg")

# Apply median blur
m3 = cv2.medianBlur(img, 3)
m5 = cv2.medianBlur(img, 5)
m7 = cv2.medianBlur(img, 7)

# Display original and filtered images
cv2.imshow("Original", img)
cv2.imshow("Median 3x3", m3)
cv2.imshow("Median 5x5", m5)
cv2.imshow("Median 7x7", m7)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Median blur → smooths the image and removes noise
# 3, 5, 7 → kernel sizes
# Larger kernel → stronger smoothing
# cv2.medianBlur() → applies median filtering
# Median filter → replaces each pixel with the median of neighbouring pixels
