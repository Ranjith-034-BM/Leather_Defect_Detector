import cv2

# load two images
img1 = cv2.imread(r'/home/ranjith/Downloads/leather.jpg')
img2 = cv2.imread(r'/home/ranjith/Downloads/leatherPiece.jpg')
img1 = cv2.resize(img1,(400,400))
img2 = cv2.resize(img2,(400,400))

# create SIFT detector object
sift = cv2.xfeatures2d.SIFT_create()

# detect keypoints and compute descriptors for both images
kp1, des1 = sift.detectAndCompute(img1, None
                                  )
kp2, des2 = sift.detectAndCompute(img2, None)

# create a BFMatcher object and match descriptors of both images
bf = cv2.BFMatcher()
matches = bf.match(des1, des2)

# draw the matches between the two images
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# display the result
cv2.imshow('Matches', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
