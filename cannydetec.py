import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('/home/ranjith/Downloads/leather.jpg')
img = cv2.resize(img,(400,400))

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform Canny edge detection with a lower and upper threshold
edges = cv2.Canny(gray, 100, 200)

# Apply Laplacian operator
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
ret,thresh = cv2.threshold(laplacian,100,250,cv2.THRESH_TRUNC)

'''
contours,hierarchy = cv2.findContours(laplacian,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(0,0,255),3)
'''

erode = cv2.erode(thresh,kernel =(5,5),iterations = 2)

# Display the result
cv2.imshow("image",img)
cv2.imshow('Laplacian', thresh)
cv2.imshow('Edges', edges)
cv2.imshow('erode', erode)


hist_erode = cv2.calcHist([gray],[0],None,[255],[0,255])
plt.figure()
plt.plot(hist_erode)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
