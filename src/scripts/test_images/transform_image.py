import cv2

img = cv2.imread("vivi.jpg")
img_resized = cv2.resize(img, [250, 250])
cv2.imshow("Image", img_resized)
cv2.imwrite("vivi_resized.jpg", img_resized)
cv2.waitKey(0)
