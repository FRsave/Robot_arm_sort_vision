import numpy as np
import cv2
import logging
import cv2
import numpy as np
import copy

# #https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
#
#
# import matplotlib.pyplot as plt
# im = plt.imread("ch.png")
# implot = plt.imshow(im)
#
# # put a blue dot at (10, 20)
# #plt.scatter([192], [801])
# #plt.scatter([937], [652])
# #plt.scatter([798], [62])
# plt.scatter([101], [197])
#
# plt.show()
#
largeTestTube = [120, 50]
smallTestTube = [120, 30]
petriDish = [100, 100]

largeTestTube[0]
largeTestTube[1]

answer = largeTestTube[0] - 20
print(answer)


#
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')
#
# def main():
#     logging.info("compute_homography.py main()")
#     image_filepath = "ch.png"
#     features_mm_to_pixels_dict =  {(0, 0): (192, 801, ),
#                                   (214.2, 0): (937, 652),
#                                   (214.2, 178.5): (798, 67),
#                                   (0, 178.5): (101, 197)}
#     A = np.zeros((2 * len(features_mm_to_pixels_dict), 6), dtype=float)
#     b = np.zeros((2 * len(features_mm_to_pixels_dict), 1), dtype=float)
#     index = 0
#     for XY, xy in features_mm_to_pixels_dict.items():
#         X = XY[0]
#         Y = XY[1]
#         x = xy[0]
#         y = xy[1]
#         A[2 * index, 0] = x
#         A[2 * index, 1] = y
#         A[2 * index, 2] = 1
#         A[2 * index + 1, 3] = x
#         A[2 * index + 1, 4] = y
#         A[2 * index + 1, 5] = 1
#         b[2 * index, 0] = X
#         b[2 * index + 1, 0] = Y
#         index += 1
#         # A @ x = b
#     x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
#
#     pixels_to_mm_transformation_mtx = np.array([[x[0, 0], x[1, 0], x[2, 0]], [x[3, 0], x[4, 0], x[5, 0]], [0, 0, 1]])
#     logging.debug("main(): pixels_to_mm_transformation_mtx = \n{}".format(pixels_to_mm_transformation_mtx))
#
#     test_xy_1 = (1399, -1599, 1) #3900
#     test_XY_1 = pixels_to_mm_transformation_mtx @ test_xy_1
#     logging.info("main(): test_XY_1 = {}; test_XY_1 = {}".format(test_xy_1, test_XY_1))
#     image = cv2.imread(image_filepath)
#     annotated_img = copy.deepcopy(image)
#     draw_crosshair(annotated_img, (round(test_XY_1[0]), round(test_XY_1[1])), 20, (255, 255, 0))
#     cv2.imwrite("annotated2.png", annotated_img)
#
#     mm_to_pixels_transformation_mtx = np.linalg.inv(pixels_to_mm_transformation_mtx)
#     test_XY_2 = (107.1, 107.1, 1)
#     test_xy_2 = mm_to_pixels_transformation_mtx @ test_XY_2
#     image = cv2.imread(image_filepath)
#     annotated_img = copy.deepcopy(image)
#     draw_crosshair(annotated_img, (round(test_xy_2[0]), round(test_xy_2[1])), 20, (0, 255, 0))
#     cv2.imwrite("annotated.png", annotated_img)
#
#
# def draw_crosshair(image, center, width, color):
#     cv2.line(image, (center[0] - width//2, center[1]), (center[0] + width//2, center[1]), color)
#     cv2.line(image, (center[0], center[1] - width//2), (center[0], center[1] + width//2), color)
#
# if __name__ == '__main__':
#     main()
# # importing the module
# import cv2

#
# # function to display the coordinates of
# # of the points clicked on the image
# def click_event(event, x, y, flags, params):
#     # checking for left mouse clicks
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # displaying the coordinates
#         # on the Shell
#         print(x, ' ', y)
#
#         # displaying the coordinates
#         # on the image window
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(img, str(x) + ',' +
#                     str(y), (x, y), font,
#                     1, (255, 0, 0), 2)
#         cv2.imshow('image', img)
#
#     # checking for right mouse clicks
#     if event == cv2.EVENT_RBUTTONDOWN:
#         # displaying the coordinates
#         # on the Shell
#         print(x, ' ', y)
#
#         # displaying the coordinates
#         # on the image window
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         b = img[y, x, 0]
#         g = img[y, x, 1]
#         r = img[y, x, 2]
#         cv2.putText(img, str(b) + ',' +
#                     str(g) + ',' + str(r),
#                     (x, y), font, 1,
#                     (255, 255, 0), 2)
#         cv2.imshow('image', img)
#
#
# # driver function
# if __name__ == "__main__":
#     # reading the image
#     img = cv2.imread('image_objects.jpg', 1)
#
#     # displaying the image
#     cv2.imshow('image', img)
#
#     # setting mouse handler for the image
#     # and calling the click_event() function
#     cv2.setMouseCallback('image', click_event)
#
#     # wait for a key to be pressed to exit
#     cv2.waitKey(0)
#
#     # close the window
#     cv2.destroyAllWindows()
#
# #
# def calculate_XYZ(self, u, v):
#     # Solve: From Image Pixels, find World Points
#
#     uv_1 = np.array([[u, v, 1]], dtype=np.float32)  #image x and y
#     uv_1 = uv_1.T #normalise it
#     suv_1 = self.scalingfactor * uv_1 #
#     xyz_c = self.inverse_newcam_mtx.dot(suv_1) #ineversed camera matrix
#     xyz_c = xyz_c - self.tvec1 #also product of pnp
#     XYZ = self.inverse_R_mtx.dot(xyz_c) # inverse of pnp result of rvec then cv2.Rodrigues(rvec1) amd then inversed
#
#
# return XYZ
#

# Code belongs to and altered for this porject:
# https://github.com/pacogarcia3/hta0-horizontal-robot-arm/blob/9121082815e3e168e35346efa9c60bd6d9fdcef1/initial_perspective_calibration.py#L10


# cam_matrix = np.array([[803.50497215 , 0.  ,       320.44270463],
#  [  0.    ,     801.88888833, 273.94703118],
#  [  0.     ,      0.      ,     1.       ]])
#
#
# dist = np.array([[ 7.14866635e-02 -7.75521953e-01,  9.60569354e-03 , 1.00233108e-03,
#    5.88020175e+00]])
#
# cx=cam_matrix[0,2]
# cy=cam_matrix[1,2]
# fx=cam_matrix[0,0]
# print("cx: "+str(cx)+",cy "+str(cy)+",fx "+str(fx))
#
# total_points_used=10
#
# X_center=10.9
# Y_center=10.7
# Z_center=43.4
# worldPoints=np.array([[X_center,Y_center,Z_center],
#                        [5.5,3.9,46.8],
#                        [14.2,3.9,47.0],
#                        [22.8,3.9,47.4],
#                        [5.5,10.6,44.2],
#                        [14.2,10.6,43.8],
#                        [22.8,10.6,44.8],
#                        [5.5,17.3,43],
#                        [14.2,17.3,42.5],
#                        [22.8,17.3,44.4]], dtype=np.float32)
#
# imagePoints=np.array([[cx,cy],
#                        [502,185],
#                        [700,197],
#                        [894,208],
#                        [491,331],
#                        [695,342],
#                        [896,353],
#                        [478,487],
#                        [691,497],
#                        [900,508]], dtype=np.float32)
#
# for i in range(1,total_points_used):
#     #start from 1, given for center Z=d*
#     #to center of camera
#     wX=worldPoints[i,0]-X_center
#     wY=worldPoints[i,1]-Y_center
#     wd=worldPoints[i,2]
#
#     d1=np.sqrt(np.square(wX)+np.square(wY))
#     wZ=np.sqrt(np.square(wd)-np.square(d1))
#     worldPoints[i,2]=wZ
#
# print(worldPoints)
#
# print("cam_matrix")
# print(cam_matrix)
# inverse_newcam_mtx = np.linalg.inv(cam_matrix)
# print("inverse matrix")
# print(inverse_newcam_mtx)
#
# print("solvePNP")
# ret, rvec1, tvec1=cv2.solvePnP(worldPoints,imagePoints,cam_matrix,dist)
#
# print("pnp rvec1 - Rotation")
# print(rvec1)
# print("pnp tvec1 - Translation")
# print(tvec1)
# print("R - rodrigues vecs")
# R_mtx, jac=cv2.Rodrigues(rvec1)
# print(R_mtx)
# print("R|t - Extrinsic Matrix")
# Rt=np.column_stack((R_mtx,tvec1))
# print(Rt)
# print("newCamMtx*R|t - Projection Matrix")
# P_mtx=cam_matrix.dot(Rt)
# print(P_mtx)
# s_arr = np.array([0], dtype=np.float32)
# s_describe = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
#
# for i in range(0, total_points_used):
#     print("=======POINT # " + str(i) + " =========================")
#
#     print("Forward: From World Points, Find Image Pixel")
#     XYZ1 = np.array([[worldPoints[i, 0], worldPoints[i, 1], worldPoints[i, 2], 1]], dtype=np.float32)
#     XYZ1 = XYZ1.T
#     print("{{-- XYZ1")
#     print(XYZ1)
#     suv1 = P_mtx.dot(XYZ1)
#     print("//-- suv1")
#     print(suv1)
#     s = suv1[2, 0]
#     uv1 = suv1 / s
#     print(">==> uv1 - Image Points")
#     print(uv1)
#     print(">==> s - Scaling Factor")
#     print(s)
#     s_arr = np.array([s / total_points_used + s_arr[0]], dtype=np.float32)
#     s_describe[i] = s
#
#
#     print("Solve: From Image Pixels, find World Points")
#
#     uv_1 = np.array([[imagePoints[i, 0], imagePoints[i, 1], 1]], dtype=np.float32)
#     uv_1 = uv_1.T
#     print(">==> uv1")
#     print(uv_1)
#     suv_1 = s * uv_1
#     print("//-- suv1")
#     print(suv_1)
#
#     print("get camera coordinates, multiply by inverse Camera Matrix, subtract tvec1")
#     xyz_c = inverse_newcam_mtx.dot(suv_1)
#     xyz_c = xyz_c - tvec1
#     print("      xyz_c")
#     inverse_R_mtx = np.linalg.inv(R_mtx)
#     XYZ = inverse_R_mtx.dot(xyz_c)
#     print("{{-- XYZ")
#     print(XYZ)
#
#     # if calculatefromCam == True:
#     #     cXYZ = cameraXYZ.calculate_XYZ(imagePoints[i, 0], imagePoints[i, 1])
#     #     print("camXYZ")
#     #     print(cXYZ)
#
# s_mean, s_std = np.mean(s_describe), np.std(s_describe)
#
# print(">>>>>>>>>>>>>>>>>>>>> S RESULTS")
# print("Mean: " + str(s_mean))
# # print("Average: " + str(s_arr[0]))
# print("Std: " + str(s_std))
#
# print(">>>>>> S Error by Point")
#
# for i in range(0, total_points_used):
#     print("Point " + str(i))
#     print("S: " + str(s_describe[i]) + " Mean: " + str(s_mean) + " Error: " + str(s_describe[i] - s_mean))