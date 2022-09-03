import cv2
import numpy as np

img = cv2.imread('image_objects.jpg')

print(img.shape)

#img = cv2.resize(img,(640,480))

cv2.circle(img, (115,10),5,(0,0,255),-1)
cv2.circle(img, (450,10),5,(0,0,255),-1)
cv2.circle(img, (20,340),5,(0,0,255),-1)
cv2.circle(img, (555,340),5,(0,0,255),-1)

width, height = 480,640  #720,1280
warp_corners = np.float32([[230,10],[900,10],[20,680],[1110,680]])
warp_corners_output = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(warp_corners,warp_corners_output)

new = cv2.warpPerspective(img,matrix,(width,height))

# for x in range (0,4):
#     cv2.circle(img,(warp_corners[x][0],warp_corners[x][1]),5,(0,0,255))

cv2.imshow("new", new)
cv2.imshow("org", img)
cv2.waitKey(50000)

