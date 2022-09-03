import cv2
import numpy as np
from PIL import Image, ImageChops
import cv2.aruco as aruco
import matplotlib as plt
import re
import math

from objectDataSave import *

loop_check = 0

list_of = []

object_number = 1

# workspace size

global workspace_x
global workspace_y

# size of markers

aruco_size = 20


# camera calibration flag

calibrate_camera = 0

# camera calibration values
#mtx = np.array([[803.50497215, 0., 320.44270463], [0., 801.88888833, 273.94703118], [0., 0., 1.]])

#dist = np.array([7.14866635e-02, -7.75521953e-01, 9.60569354e-03, 1.00233108e-03, 5.88020175e+00])



mtx = np.array(
[[1.89129278e+03, 0.00000000e+00, 1.46899834e+03],
 [0.00000000e+00, 1.86324300e+03, 8.26025303e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist = np.array([-0.35883596,  0.16373543,  0.00161949,  0.00222944, -0.04379734])







#mtx = np.array([[1.66193377e+03, 0.00000000e+00, 1.25207306e+03],[0.00000000e+00, 1.69806678e+03, 1.11202446e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

#dist = np.array([-0.34737502,  0.18058007, -0.02705915,  0.00730897, -0.0581378 ])




def init_new_json_file():
    global object_number
    object_number = cX = cY = p1 = p2 = p3 = p4 = angle = w = h = object_type = "-"
    object = Object(object_number, cX, cY, p1, p2, p3, p4, angle, w, h, object_type )
    object.save_to_json(filename)
    object_number = 1


# function to seperte x,y of objects from objects
def jsonReadTest():
    j = open("objects.json", 'r')
    data = j.read()
    obj = json.loads(data)
    list = obj['objects']
    e = list[3].get("p1")
    e = e.split(" ")

    x = e[0]
    y = e[1]
    x = x.replace("[", "")
    y = y.replace("]", "")
    int(x)
    int(y)

    print(x, y)


# test for data read for ROS side
# with this, we can iterate actions per number of elements in Json file

def jsonReadTest2():
    j = open("objects.json", 'r')
    data = j.read()
    obj = json.loads(data)
    list = obj['objects']

    num_of_objects = len(list)

    print(len(list))

    for i in range(num_of_objects):
        if i != 0:
            cx = list[i].get("centre X")
            cy = list[i].get("centre Y")

            int(cx)
            int(cy)

            print(cx, cy)


def specify_object():
    j = open("objects.json", 'r')
    data = j.read()
    obj = json.loads(data)
    list = obj['objects']
    print(list)

    usr_inp = int(input("Specify object to focus on (enter single digit) \n"))

    print(list[usr_inp])


# def classify_by_size(w,h):
#   global test_tube_large, test_tube_small, petri_dish

#  if


#   return obj_type


# def shape_class(c, x, y, w, h):
#     shape_estimation = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
#
#     # circle = cv2.HoughCircles()
#
#     if len(shape_estimation) == 4:
#         ratio = float(w) / h
#         print(ratio)
#
#         # aspect ratio to determine when a rectangle is close to be a square
#         if ratio >= 0.80 and ratio <= 1.40:
#             print("square")
#             obj_shape = str("square")
#         else:
#             print("rectangle")
#             obj_shape = str("rectangle")
#
#     #
#
#     elif len(shape_estimation) > 8:
#
#         obj_shape = str("circle")
#
#     return obj_shape
def object_by_size(line1_l, line2_l):

    # Large test tube = 120mm and 50mm
    # small test tube = 120mm and 30mm
    # petri dish      = 100mm and 100mm

    lTestTube = [135,50]
    sTestTube = [120,30]
    petriDish  = [100, 100]
    penlid = [80, 30]

    #range of values between
    r = 25
    #r_ltube = 35


    line1_l = int(line1_l)
    line2_l = int(line2_l)

    # data change here to decreased if statement length
    penL1 = int(penlid[0])
    penL2 = int(penlid[1])

    stube1 = int(sTestTube[0])
    stube2 = int(sTestTube[1])

    ltube1 = int(lTestTube[0])
    ltube2 = int(lTestTube[1])

    pdish1 = int(petriDish[0])
    pdish2 = int(petriDish[1])



    if (line1_l in range(penL1 - r, penL1 + r) and line2_l in range(penL2 - r,penL2 + r)) or\
            (line1_l in range(penL2 - r, penL2 + r) and line2_l in range(penL1 - r, penL1 +r)):
        print("object is a pen lid")
        object_type = "pen lid"


    elif (line1_l in range(stube1 -r,stube1 +r)
        and line2_l in range(stube2-r,stube2+r)) \
            or (line1_l in range(stube2-r,stube2+r)
                and line2_l in range(stube1 -r,stube1 +r)):

        print("object is a small test tube")
        object_type = "small test tube"

    elif (line1_l in range(ltube1 -r,ltube1+r)
        and line2_l in range(ltube2 -r,ltube2+r)) \
            or (line1_l in range(ltube2 -r,ltube2+r)
                and line2_l in range(ltube1 -r,ltube1+r)):

        print("object is a large test tube")
        object_type = "large test tube"

    elif (line1_l in range(pdish1 - r, pdish1 + r)
          and line2_l in range(pdish2 - r, pdish2 + r)) \
            or (line1_l in range(pdish2 - r, pdish2 + r)
                and line2_l in range(pdish1 - r, pdish1 + r)):

        print("object is a petri dish")
        object_type = "petri dish "

    else:
        print("unknown object")
        object_type = "unknown"



    return object_type






# turing position coordinates to line length for size approximation
def size_comparison(p1, p2, p3):

    # distnace between two lines
    line1_l = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    line2_l = math.sqrt((p2[0] - p3[0]) ** 2 + (p3[1] - p2[1]) ** 2)

    print("line1 is: " + str(line1_l) +" line2 is:" + str(line2_l))

    return line1_l, line2_l


# 3D to 2D coords transformation formula:
# https://github.com/sebastiengilbert73/tutorial_homography/blob/main/compute_homography.py
# https://towardsdatascience.com/coordinates-of-a-flat-scene-b37487df63ca


def correct_image(img):
    global mtx, dist

    h, w = img.shape[:2]
    w1, h1 = 1 * w, 1 * h

    #print(w, h)

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w1, h1))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # crop the image x,y,w,h = roi dst = dst[y:y+h, x:x+w]
    x, y, w, h = roi
   # print(w, h)

    dst = dst[y:y + h, x:x + w]

    return dst


# return


def _2D_3D_transf(valx, valy, c1, c2, c3, c4):
    global workspace_x, workspace_y, p1_1, p1_2, p2_1, p2_2, p3_1, p3_2, p4_1, p4_2, arIds, arCorner


    corners = arCorner
    ids = arIds
    print("marker count: " + str(len(ids)))

    # stores centre of markers, upper left clock wise motion
    cent = np.empty((4, 2))
    for i, c in zip(ids.ravel(), corners):
        cent[i] = c[0].mean(axis=0)

    n = ids

    for i in range(len(n)):

        g = cent[i].round()

        x, y = np.array_split(g, 2)

        x = str(x)[1:-1]
        y = str(y)[1:-1]

        if i == 0:
            p1_1, p1_2 = x, y
        elif i == 1:
            p2_1, p2_2 = x, y
        elif i == 2:
            p3_1, p3_2 = x, y
        elif i == 3:
            p4_1, p4_2 = x, y

    #print(p1_1, p1_2, p2_1, p2_2)

    int(float(p1_1))
    int(float(p1_2))
    int(float(p2_1))
    int(float(p2_2))
    int(float(p3_1))
    int(float(p3_2))
    int(float(p4_1))
    int(float(p4_2))

    #print(p1_1, p1_2)
    workspace_x = int(workspace_x)
    workspace_y = int(workspace_y)

    # print("values in:")
    # print(p4_1, p4_2)
    # print(p1_1,p1_2)

    # real world______:_____pixels
    features_mm_to_pixels_dict = {(0, 0): (p4_1, p4_2),
                                  (workspace_x, 0): (p3_1, p3_2),
                                  (workspace_x, workspace_y): (p2_1, p2_2),
                                  (0, workspace_y): (p1_1, p1_2)}

   # print(features_mm_to_pixels_dict)

    A = np.zeros((2 * len(features_mm_to_pixels_dict), 6), dtype=float)
    b = np.zeros((2 * len(features_mm_to_pixels_dict), 1), dtype=float)

    index = 0
    for XY, xy in features_mm_to_pixels_dict.items():
        X = XY[0]
        Y = XY[1]
        x = xy[0]
        y = xy[1]
        A[2 * index, 0] = x
        A[2 * index, 1] = y
        A[2 * index, 2] = 1
        A[2 * index + 1, 3] = x
        A[2 * index + 1, 4] = y
        A[2 * index + 1, 5] = 1
        b[2 * index, 0] = X
        b[2 * index + 1, 0] = Y
        index += 1
    # A @ x = b
    x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)

    pixels_to_mm_transformation_mtx = np.array([[x[0, 0], x[1, 0], x[2, 0]], [x[3, 0], x[4, 0], x[5, 0]], [0, 0, 1]])

    # print(c4)
    # print(type(c1))
    # print(type(c4))

    number1 = re.findall(r'\d+', c1)
    c1_x =(number1[0])
    c1_y =(number1[1])

    number2 = re.findall(r'\d+', c2)
    c2_x =(number2[0])
    c2_y =(number2[1])

    number3 = re.findall(r'\d+', c3)
    c3_x =(number3[0])
    c3_y =(number3[1])

    number4 = re.findall(r'\d+', c4)
    c4_x =(number4[0])
    c4_y =(number4[1])



    c1 = ((float(c1_x)),(float(c1_y)), 1)

    c2 = ((float(c2_x)), (float(c2_y)), 1)

    c3 = ((float(c3_x)), (float(c3_y)), 1)

    c4 = ((float(c4_x)), (float(c4_y)), 1)


    # print("============")
    # print(c1)
    # print("============")



    valx = valx
    valy = valy

    #print(c1,c2,c3,c4)

    test_xy_1 = (valx, valy, 1)
    test_XY_1 = pixels_to_mm_transformation_mtx @ test_xy_1

    newC1 = pixels_to_mm_transformation_mtx @ c1
    newC2 = pixels_to_mm_transformation_mtx @ c2
    newC3 = pixels_to_mm_transformation_mtx @ c3
    newC4 = pixels_to_mm_transformation_mtx @ c4

    new_valx = test_XY_1[0]
    new_valy = test_XY_1[1]

    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print(test_XY_1)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print(new_valy)

    #print(new_valx, new_valy)

    # img_marked = aruco.drawDetectedMarkers(img, corners, ids)
    # img_marked = cv2.circle(img_marked, (val, val1), 5, (255, 125, 255), -1)
    # cv2.imshow("w", img_marked)
    # cv2.waitKey(60000)

    #print(newC1)
    # print("workspace size")
    # print(workspace_x, workspace_y)
    # print("real corner values")
    # print(newC1, newC2, newC3, newC4)

    return new_valx, new_valy , newC1, newC2, newC3,newC4


#    mm_to_pixels_transformation_mtx = np.linalg.inv(pixels_to_mm_transformation_mtx)


def try_again():
    usr_inp = int(input("Would you like to try again or use this data?\n Press 1 to try again from start \n Press 2 to re-do "
              "from object addtion \n Press 3 to exit\n"))

    #jsonReadTest()

    if usr_inp == 1:
        image_intro()
    elif usr_inp == 2:
        adding_objects()
    elif usr_inp == 3:
        exit()
    else:
        print("invalid input...\n try again!\n")
        try_again()


def detect_object():
    global object_number

    orginal_base = str('image_base.jpg')
    orginal_objects = str('image_objects.jpg')

    img1 = Image.open(orginal_base)
    img2 = Image.open(orginal_objects)

    diff = ImageChops.difference(img1, img2)

    # aruco_image_flatten(orginal_base)
    # aruco_image_flatten(orginal_objects)

    if diff.getbbox():
        # diff.show()

        # convert image into array for openCV
        img4 = np.asarray(diff)

    image_gray = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)

    # cv2.imshow("mass", image_gray)

    lower = np.array([80, 81, 81])  # 79 is optimim(ish)
    higher = np.array([255, 255, 255])  # 186

    mask = cv2.inRange(image_gray, lower, higher)
    mask = cv2.blur(mask, (4, 4))
    cv2.imshow("mask", mask)

    cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cont = sorted(cont, key=cv2.contourArea, reverse=True)

    # cont = sorted(cont, key=lambda x:cv2.boundingRect(x)[0])

    new = cv2.drawContours(image_gray, cont, -1, (0, 255, 0), 2);
    cv2.imshow("2", new)
    print("number of cnt", len(cont))

    for (i, c) in enumerate(cont):

        # alternative method that draws around contours
        # epsilon = 0.01* cv2.arcLength(cnt,True)
        # approx = cv2.approxPolyDP(cnt, epsilon, True)
        # img4 = cv2.drawContours(img4, [approx], 0, (0, 255, 0), 2)

        # mx = max(cont, key=cv2.contourArea)

        #  print("cY", cY)
        #  print("cX", cX)

        x, y, h, w = cv2.boundingRect(c)

        # altert values to specify size -------------------£££££££££££££££££££££££££££££££££££££££££

        if (w * h) >= 1000:

            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                # set values as what you need in the situation
                cX, cY = 0, 0

            print("object: ", object_number, "object centre is at: ", c[0])

            #shape_class(c, x, y, w, h)

            # cv2.rectangle(img4, (x, y), (x + w, y + h), (0, 255, 5), 5)
            rect = cv2.minAreaRect(c)

            box = cv2.boxPoints(rect)
            # print(box)
            box = np.int0(box)
            cv2.drawContours(img4, [box], 0, (0, 255, 0), 1)
            cv2.circle(img4, (cX, cY), 5, (255, 255, 255), -1)
            cv2.putText(img4, "object:" + str(object_number), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            cv2.circle(img4, (cX, cY), 5, (255, 255, 255), -1)

            angle = rect[-1]

            # print(h)
            # print(w)

            # h  =  rect[-2]

            #print(h)

            if angle < -45:
                angle = -(90 + angle)

            else:
                angle = -angle

            #print(angle)

            #print("''''''''''''")

            p1 = box[0]
            p1 = str(p1)
            p1.strip("[]")

            p2 = box[1]
            p2 = str(p2)
            p2.strip("[]")

            p3 = box[2]
            p3 = str(p3)
            p3.strip("[]")

            p4 = box[3]
            p4 = str(p4)
            p4.strip("[]")

            for (x, y) in box:
                cv2.circle(img4, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.putText(img4, str((int(x), int(y))), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

            #print("11111111")
            #print(p1, p2)
            #print("The variable, lessons is of type:", type(p1))

            realX, realY, p1, p2, p3, p4 = _2D_3D_transf(cX, cY, p1, p2, p3, p4)

            line1, line2 = size_comparison(p1, p2, p3)

            object_type = object_by_size(line1,line2)


            object = Object(object_number, cX, cY, p1, p2, p3, p4, angle, w, h, object_type)
            print("filesave...")
            # object.print_info()
            # object.save_to_json("objects.json")
            # change image to 3 dimensional position prediction1

            p1 = np.array_str(p1, precision=2,suppress_small=True)
            p2 = np.array_str(p2, precision=2,suppress_small=True)
            p3 = np.array_str(p3, precision=2,suppress_small=True)
            p4 = np.array_str(p4, precision=2,suppress_small=True)


            object.add_to_file("objects.json", object_number, realX, realY, p1, p2, p3, p4, angle, w, h, object_type)
            print("===================================================")

            # object counter for the session (starts at 1)
            object_number = object_number + 1

    cv2.imwrite('results.jpg', img4)
    cv2.imshow("Final Image", img4)
    #cv2.imshow('image_objects.jpg')

    if cv2.waitKey(1) & 0XFF == ord('q'):
        cv2.destroyAllWindows()

    try_again()


# ArUco marker code inspired from:
# https://www.linuxtut.com/en/c6e468da7007734c897f/


def aruco_image_flatten(image_name):
    global workspace_x, workspace_y, arCorner, arIds

    n = 0

    print(str(image_name))

    name_of_file = str(image_name)

    Image_to_flatten = Image.open(image_name)

    aruco = cv2.aruco
    p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    img = cv2.imread(name_of_file)  # image_objects.jpg
    # cv2.imshow("Image check", img)
    # cv2.waitKey(5)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, p_dict)  # detection

    if ids is not None:
        n = len(ids)
        print(n)

    if n == 4:
        img = cv2.imread(name_of_file)
        img_marked = aruco.drawDetectedMarkers(img.copy(), corners, ids)  # Overlay detection results

        #print(ids)
        cv2.imshow("w", img_marked)

        # print(corners)# for testing purposes (4 corners are necessary to be functional)

        # Store the "center coordinates" of the marker in m in order from the upper left in the clockwise direction.
        m = np.empty((4, 2))
        for i, c in zip(ids.ravel(), corners):
            m[i] = c[0].mean(axis=0)

        width, height = (500, 500)  # Image size after transformation

        marker_coordinates = np.float32(m)
        true_coordinates = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        trans_mat = cv2.getPerspectiveTransform(marker_coordinates, true_coordinates)
        img_trans = cv2.warpPerspective(img, trans_mat, (width, height))

        cv2.imshow("XX", img_trans)

        tmp = img_trans.copy()

        detected_obj = list()  # Storage destination of detection result
        tr_x = lambda x: x * workspace_x / 500  # X-axis image coordinates → real coordinates
        tr_y = lambda y: y * workspace_y / 500  # Y axis
        img_trans_marked = img_trans.copy()

        cv2.imwrite(str(image_name), img_trans_marked)

        cv2.imshow("d", img_marked)  # display
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    elif n < 4:
        print("Less then 4 markers in image\n Morph image Failed!")

    else:
        print("undesired number of markers detected: 4 markers of type (aruco.DICT_4X4_(i)) required! ")

    return


def adding_objects():

    global workspace_x, workspace_y, arCorner ,arIds ,calibrate_camera



    usr_inp6 = input("Add item to the workspace! \n Enter 1 when ready...\n (2 to exit)\n")


    if usr_inp6 == "1":



        print("Taking the Image...\n")
        cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY, params=[
            cv2.CAP_PROP_FRAME_WIDTH, 1280,
            cv2.CAP_PROP_FRAME_HEIGHT, 720])



        focus = 0  # min: 0, max: 255, increment:5
        cap.set(28, focus)
        # video capture source camera (Here webcam of laptop)
        ret, frame = cap.read()  # return a single frame in variable `frame`

        # remove camera distortion
        if int(calibrate_camera) == 1:
            print("correcting image")
            frame = correct_image(frame)

        aruco = cv2.aruco
        p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        # cv2.imshow("Image check", img)1
        # cv2.waitKey(5)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame.copy(), p_dict)  # detection


        arCorner = corners
        arIds = ids
        if ids is not None:
            n = len(ids)
            print("numer of markers in image = ", n)
            img_marked = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
            if len(ids) < 4:
                aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
                parameters = aruco.DetectorParameters_create()
                #
                corners, ids, rejectedImgPoints = aruco.detectMarkers(
                    img_marked, aruco_dict, parameters=parameters)
                cv2.imshow("w", img_marked)
                cv2.waitKey(10)
                print("Less then 4 markers in image")

        cv2.imshow('object image', frame)
        cv2.waitKey(6)

        usr_inp3 = input("Is the image satisfactory? y/n \n")

        if usr_inp3 == "y":
            cv2.imwrite('image_objects.jpg', frame)
            cv2.destroyAllWindows()
            cap.release()

            detect_object()

        elif usr_inp3 == "n":
            adding_objects()
    elif usr_inp6 == 2:
        exit()





def take_pic():

    global calibrate_camera, workspace_x, workspace_y



    cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY, params=[cv2.CAP_PROP_FRAME_WIDTH, 1280,cv2.CAP_PROP_FRAME_HEIGHT, 720])#
    focus = 0  # min: 0, max: 255, increment:5
    cap.set(28, focus)

    # video capture source camera
    ret, frame = cap.read()

    # remove camera distortion

    print(calibrate_camera)

    if int(calibrate_camera) == 1:
        print("correcting image")
        frame = correct_image(frame)


    cv2.imshow('img1', frame)
    cv2.waitKey(6)

    aruco = cv2.aruco
    p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    # cv2.imshow("Image check", img)
    # cv2.waitKey(5)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, p_dict)  # detection

    if ids is not None:
        n = len(ids)
        print("numer of markers in image = ", n)
        img_marked = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        if len(ids) < 4:
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters_create()
            #
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                img_marked, aruco_dict, parameters=parameters)
            cv2.imshow("w", img_marked)
            cv2.waitKey(6)
            print("Less then 4 markers in image")
    elif ids is None:
        print(
            "No markers detected in the image!\nScaling will fail if markers are not used! Add markers and try again.")

    usr_inpiut2 = input("Is the image satisfactory? y/n \n")

    if usr_inpiut2 == "y":

        cv2.imwrite('image_base.jpg', frame)
        cv2.destroyAllWindows()
        cap.release()

        print("Done...")

        print("Workspace length and width are required in relation to camera POV")
        usr_inpWSy = input("Enter width(mm)  X (horizontal distance from the camera between centres of two markers"
                           "(longest distance)\n")
        usr_inpWSx = input("Enter length(mm) Y (Vertical distance from the camera between centres of two markers)\n")

        workspace_x = int(usr_inpWSx)
        workspace_y = int(usr_inpWSy)

        print("Width = " + str(workspace_x) + "mm")
        print("Length= " + str(workspace_y) + "mm")

        adding_objects()

        # detect_object()




    elif usr_inpiut2 == "n":

        usr_inpiut3 = input("Try again? (y/n) \n ")

        if usr_inpiut3 == "y":
            take_pic()
        elif usr_inpiut3 == "n":
            print("Stopping...")
            exit()

    else:
        print("invalid input")
        take_pic()


def image_intro():
    global calibrate_camera
    print(
        "----------------------------------------------------------------------------------------------------------\n")
    print("Welcome to object detection!\n ")
    print("We recommend to measure horizontal and vertical distance between two the markers before process begins")
    print("Measure from centre of one marker to another centre, note down the x and y for later\n")

    usr_inp_calib = input("Would you like to use camera calibration: For Yes enter 1, for No enter 0\n")
    calibrate_camera = usr_inp_calib

    usr_input = int(input("Select option to begin: \n\n  Workspace layout image 1: Press 1 \n  Clear json: Press 2 \n"))

    # 1print(usr_input,"\n")

    if usr_input == 1:
        take_pic()

    elif usr_input == 2:
        init_new_json_file()
        print("json file reset with formatting ...\n")
        image_intro()

    else:
        print("invalid input")
        image_intro()


if __name__ == '__main__':
   # img = cv2.imread("test_corners.png")
   # wa, w2 = _2D_3D_transf(img, 240, 460)
   #  print(wa)
   #  print(w2)
    # jsonReadTest2()
    image_intro()


# def take_pic():
#     print("hi")
#     cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop)
#     ret,frame = cap.read() # return a single frame in variable `frame`
#
#     while(True):
#         cv2.imshow('img1',frame)
#         cv2.waitKey(1)
#         cv2.imwrite('image-base.png',frame)
#         cv2.destroyAllWindows()
#         break
#
#     cap.release()
#
#     view_image = cv2.imread('image-base.png',1)
#
#     cv2.imshow("", view_image)
#
#     usr_inpiut2 = input("Is the image satisfactory?")
#
#     if usr_inpiut2 == "y":
#         print("works")
#     elif usr_inpiut2 == "n":
#         print("try again?")
#     else:
#         print("invalid input")
