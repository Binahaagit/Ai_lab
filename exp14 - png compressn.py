import cv2
import os

img = cv2.imread("input.jpg")
# PNG compression levels
levels = [0, 3, 5, 7, 9]

print("Compression\tFile Size(KB)")

for l in levels:
    filename = f"image_c{l}.png"
    cv2.imwrite(filename, img, [cv2.IMWRITE_PNG_COMPRESSION,l])

    size = os.path.getsize(filename) / 1024

    print(l,"\t\t",round(size, 2))
  
# Sample Output:
# Compression    File Size(KB)
# 0              856.42
# 3              245.18
# 5              214.67
# 7              207.31
# 9              201.54

# PNG compression → reduces image file size
# PNG is a lossless compression format
# Compression levels → 0, 3, 5, 7, 9
# Level 0 → no compression
# Level 9 → maximum compression
# Higher compression → usually smaller file size
# Image quality remains the same because PNG is lossless
# cv2.imwrite() → saves the compressed image
# cv2.IMWRITE_PNG_COMPRESSION → sets PNG compression level
# os.path.getsize() → gets file size in bytes
# Divide by 1024 → converts bytes to KB
