import cv2
img = cv2.imread("input.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#Negative transformn: s=255-r
neg_binary = 255 - binary
neg_gray = 255 - gray
neg_clr = 255 - img

# Display
cv2.imshow("Binary", binary)
cv2.imshow("Negative Binary", neg_binary)

cv2.imshow("Grayscale", gray)
cv2.imshow("Negative Grayscale", neg_gray)

cv2.imshow("Colour", img)
cv2.imshow("Negative Colour", neg_clr)

cv2.waitKey(0)
cv2.destroyAllWindows()
