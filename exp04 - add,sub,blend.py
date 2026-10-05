import cv2

img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")

# Make both images the same size
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

added = cv2.add(img1, img2)
subtracted = cv2.subtract(img1, img2)

blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)  #50% of image1 blend with 50% of img2(alter 0.5 if needed)

cv2.imshow("Addition", added)
cv2.imshow("Subtraction", subtracted)
cv2.imshow("Blending", blended)

cv2.waitKey(0)
cv2.destroyAllWindows()


#resize -> add -> subtract -> addWeighted
#Addnl info:😌
#blend formula = α × img1 + β × img2 + γ 
#So here, 0.5 × img1 + 0.5 × img2 + 0 (0 is gamma value to adjust brightness)
