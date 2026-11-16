
import cv2
import numpy as np

img = cv2.imread("input.jpg")

h, w, c = img.shape
output = np.zeros(((h-3)//2+1, (w-3)//2+1, 3), np.uint8)

for ch in range(3):
    for i in range(0, h-2, 2):
        for j in range(0, w-2, 2):

            region = img[i:i+3, j:j+3, ch]
            output[i//2, j//2, ch] = np.mean(region)  #use mean

cv2.imshow("Average Pooling", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
