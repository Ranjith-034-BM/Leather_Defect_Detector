import cv2
import numpy as np

# Load the image
img = cv2.imread("C:/Users/ELCOT/Documents/toe leath.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to the image
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours in the image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the image
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# Calculate the surface area of the largest contour
max_contour = max(contours, key=cv2.contourArea)
area = cv2.contourArea(max_contour)

# Display the image with the contours and area
cv2.imshow("Image with contours", img)
print("Surface area of object in image is", area)

cv2.waitKey(0)
cv2.destroyAllWindows()
