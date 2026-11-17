# Q2 - Given an original image and a template image, perform template matching using Mean Squared Error (MSE).
# Find the best matching location of the template in the original image and draw a rectangle around the matched region.

## TEMPLATE MATCHING USING MSE (MEAN SQUARED ERROR) ##

# ============================================================
# CODE 1: USING OPENCV BUILT-IN FUNCTION
# ============================================================

import cv2

# Read original image and template in grayscale
img = cv2.imread("image.jpg", 0)
temp = cv2.imread("template.jpg", 0)

# Get template height and width
h, w = temp.shape

# Perform template matching using Squared Difference
# TM_SQDIFF gives minimum error for the best match
result = cv2.matchTemplate(img, temp, cv2.TM_SQDIFF)

# Find minimum and maximum values and their locations
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Minimum error location gives the best matching position
x, y = min_loc

# Draw rectangle around the matched region
# (x, y) = top-left corner
# (x+w, y+h) = bottom-right corner
# 255 = White color, 2 = Thickness
cv2.rectangle(img, (x, y), (x+w, y+h), 255, 2)

# Display result
cv2.imshow("Best Match", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Minimum Error:", min_val)
print("Best Match Position:", min_loc)


# IMPORTANT POINTS - CODE 1:
# 1. cv2.imread("image.jpg", 0) reads image in grayscale.
# 2. temp.shape returns template dimensions.
# 3. cv2.matchTemplate() compares template with original image.
# 4. cv2.TM_SQDIFF uses squared difference.
# 5. cv2.minMaxLoc() finds minimum/maximum values and locations.
# 6. min_loc gives the best matching position.
# 7. cv2.rectangle() highlights the matched region.
# TM_SQDIFF -> Minimum value is best.
# TM_CCOEFF -> Maximum value is best.
# TM_CCORR  -> Maximum value is best.


# ============================================================
# CODE 2: USING NUMPY ARRAY MANIPULATION (MANUAL MSE)
# ============================================================

import cv2
import numpy as np

# Read original image and template in grayscale
img = cv2.imread("image.jpg", 0)
temp = cv2.imread("template.jpg", 0)

# Get template dimensions
h, w = temp.shape

# Initialize minimum error with infinity
min_error = float('inf')

# Initialize best matching position
best_x, best_y = 0, 0

# Slide template over original image
# img.shape[0] = Image height
# img.shape[1] = Image width
for y in range(img.shape[0] - h + 1):
    for x in range(img.shape[1] - w + 1):

        # Extract a region having the same size as template
        # NumPy slicing: rows first, columns second
        window = img[y:y+h, x:x+w]

        # Calculate Mean Squared Error (MSE)
        # Convert to float to avoid uint8 subtraction problems
        # Subtract corresponding pixels, square and calculate mean
        error = np.mean((window.astype(float) - temp.astype(float)) ** 2)

        # Update minimum error and best matching position
        if error < min_error:
            min_error = error
            best_x = x
            best_y = y

# Draw rectangle around the best matching region
cv2.rectangle(img, (best_x, best_y),
              (best_x+w, best_y+h), 255, 2)

# Display result
cv2.imshow("Best Match", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Minimum MSE:", min_error)
print("Best Match Position:", (best_x, best_y))


# IMPORTANT POINTS - CODE 2:
# 1. temp.shape gives template height and width.
# 2. img.shape[0] gives original image height.
# 3. img.shape[1] gives original image width.
# 4. float('inf') initializes minimum error with infinity.
# 5. Outer loop moves the template vertically.
# 6. Inner loop moves the template horizontally.
# 7. img[y:y+h, x:x+w] extracts a region using NumPy slicing.
# 8. astype(float) avoids unsigned integer subtraction problems.
# 9. np.mean() calculates the average squared difference.
# 10. if error < min_error updates the best matching position.
# Minimum MSE = Best matching position.


# ============================================================
# MSE FORMULA
# ============================================================

# MSE = SUM((I1 - I2)^2) / (M * N)
#
# Subtract corresponding pixels.
# Square the differences.
# Calculate their mean.


# ============================================================
# QUICK EXAM REVISION
# ============================================================

# CODE 1:
# matchTemplate -> minMaxLoc -> rectangle

# CODE 2:
# Slicing -> Subtraction -> Squaring -> Mean
# -> Minimum error -> Rectangle

# TM_SQDIFF: Minimum value is best.
# TM_CCOEFF: Maximum value is best.
# TM_CCORR: Maximum value is best.
