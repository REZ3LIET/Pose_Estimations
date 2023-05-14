import cv2 as cv

img = cv.imread("E:\Github\Advanced_CV\Pose_Estimations\Body_Estimation\Data\Images\humans_2.jpg")
# print(img.shape)
img = cv.resize(img, (640, 480))
cv.imshow("Image", img)
cv.waitKey(0)
cv.destroyAllWindows()