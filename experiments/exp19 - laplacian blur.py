import cv2

# Read image as grayscale
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Laplacian filter
lap3 = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
lap5 = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
lap7 = cv2.Laplacian(img, cv2.CV_64F, ksize=7)

# Convert results to 8-bit for display
lap3 = cv2.convertScaleAbs(lap3)
lap5 = cv2.convertScaleAbs(lap5)
lap7 = cv2.convertScaleAbs(lap7)

# Display images
cv2.imshow("Original", img)
cv2.imshow("Laplacian 3x3", lap3)
cv2.imshow("Laplacian 5x5", lap5)
cv2.imshow("Laplacian 7x7", lap7)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Laplacian filter → detects edges in an image
# Grayscale image is used for Laplacian filtering
# 3, 5, 7 → kernel sizes
# cv2.Laplacian() → applies the Laplacian filter
# CV_64F → safely handles positive and negative values
# convertScaleAbs() → converts result to 8-bit for display

# Original image
#      ↓
# Laplacian calculation
#      ↓
# CV_64F → safely handles negative values
#      ↓
# convertScaleAbs()
#      ↓
# 8-bit image (0–255)
#      ↓
# Display
