import cv2
import cv2.aruco as aruco
import numpy as np

#https://www.linuxtut.com/en/c6e468da7007734c897f/

#Marker generation


# aruco = cv2.aruco
#
# p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
# marker =  [0] * 4 #Initialization
# for i in range(len(marker)):
#   marker[i] = aruco.drawMarker(p_dict, i, 550) # 75x75 px
#   cv2.imwrite(f'marker{i}.png', marker[i])

p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
img = cv2.imread('image_objects.jpg')
corners, ids, rejectedImgPoints = aruco.detectMarkers(img, p_dict) #detection
img_marked = aruco.drawDetectedMarkers(img.copy(), corners, ids)   #Overlay detection results

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

corners, ids, rejectedImgPoints = aruco.detectMarkers(
img_marked, aruco_dict, parameters=parameters)
print(corners)



aruco = cv2.aruco
p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
img = cv2.imread('image_objects.jpg')#image_objects.jpg
cv2.imshow("z", img)
cv2.waitKey(5)

corners, ids, rejectedImgPoints = aruco.detectMarkers(img, p_dict) #detection

print(corners)

#Store the "center coordinates" of the marker in m in order from the upper left in the clockwise direction.
m = np.empty((4,2))
for i,c in zip(ids.ravel(), corners):
  m[i] = c[0].mean(axis=0)

width, height = (500,500) #Image size after transformation

marker_coordinates = np.float32(m)
true_coordinates   = np.float32([[0,0],[width,0],[width,height],[0,height]])
trans_mat = cv2.getPerspectiveTransform(marker_coordinates,true_coordinates)
img_trans = cv2.warpPerspective(img,trans_mat,(width, height))
cv2.imshow("XX",img_trans)

tmp = img_trans.copy()




distance_of_markers_1 = 150
distance_of_markers_2 = 23

detected_obj = list() #Storage destination of detection result
tr_x = lambda x : x * distance_of_markers_1 / 500 #X-axis image coordinates → real coordinates
tr_y = lambda y : y * distance_of_markers_2 / 500 #Y axis 〃
img_trans_marked = img_trans.copy()


cv2.imshow("w",img_trans_marked)

cv2.imshow("d",img_marked) #display
cv2.waitKey(5000)