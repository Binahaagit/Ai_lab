import cv2

img = cv2.imread("input.jpg")

# Apply Gaussian Blur
g3 = cv2.GaussianBlur(img, (3, 3), 0)
g5 = cv2.GaussianBlur(img, (5, 5), 0)
g7 = cv2.GaussianBlur(img, (7, 7), 0)

# Display original and filtered images
cv2.imshow("Original", img)
cv2.imshow("Gaussian 3x3", g3)
cv2.imshow("Gaussian 5x5", g5)
cv2.imshow("Gaussian 7x7", g7)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Gaussian blur → smooths the image and reduces noise
# (3,3), (5,5), (7,7) → kernel sizes
# Larger kernel → stronger smoothing
# sigmaX = 0 → automatically calculated by OpenCV
