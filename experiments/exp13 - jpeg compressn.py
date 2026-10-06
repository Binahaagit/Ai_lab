import cv2
import os

# Read image
img = cv2.imread("input.jpg")
qualities = [90, 70, 50, 30, 10]    # JPEG quality levels

print("Quality\tFile Size (KB)")

for q in qualities:
    filename = f"image_q{q}.jpg"
    cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, q])

    size = os.path.getsize(filename) / 1024

    print(q, "\t",round(size, 2))

# JPEG compression → reduces image file size
# JPEG is a lossy compression format
# Quality levels → 90, 70, 50, 30, 10
# Higher quality → better image quality and usually larger file size
# Lower quality → lower image quality and usually smaller file size
# cv2.imwrite() → saves the compressed image
# cv2.IMWRITE_JPEG_QUALITY → sets JPEG quality
# os.path.getsize() → gets file size in bytes
# Divide by 1024 → converts bytes to KB
