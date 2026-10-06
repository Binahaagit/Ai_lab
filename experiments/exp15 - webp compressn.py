import cv2
import os

img = cv2.imread("input.jpg")

qualities = [100, 70, 50, 30, 10]

print("Quality\tFile Size(KB)")

for q in qualities:
    filename = f"image_q{q}.webp"
    cv2.imwrite(filename, img, [cv2.IMWRITE_WEBP_QUALITY, q])

    size = os.path.getsize(filename) / 1024

    print(q, "\t", round(size, 2))

# Sample Output:
# Quality    File Size(KB)
# 100        125.42
# 70         52.31
# 50         38.76
# 30         28.45
# 10         18.21

# WEBP compression → reduces image file size
# WEBP → image format designed for good quality and smaller file size
# WEBP supports both lossy and lossless compression
# WEBP quality levels → 100, 70, 50, 30, 10
# Higher quality → better image quality and usually larger file size
# Lower quality → lower image quality and usually smaller file size
# cv2.IMWRITE_WEBP_QUALITY → sets WEBP quality
# os.path.getsize() → gets file size in bytes
# Divide by 1024 → converts bytes to KB
