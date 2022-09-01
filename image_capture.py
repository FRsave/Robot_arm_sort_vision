import cv2
import numpy as np
from PIL import Image, ImageChops
import cv2.aruco as aruco
import matplotlib as plt

from objectDataSave import *

list_of = []

object_number = 1

# workspace size

workspace_x = 150
workspace_y = 150

# size of markers

aruco_size = 20


# camera calibration flag

calibrate_camera = 0

# camera calibration values
mtx = np.array([[803.50497215, 0., 320.44270463], [0., 801.88888833, 273.94703118], [0., 0., 1.]])

dist = np.array([7.14866635e-02, -7.75521953e-01, 9.60569354e-03, 1.00233108e-03, 5.88020175e+00])


def init_new_json_file():
    object_number = cX = cY = p1 = p2 = p3 = p4 = angle = w = h = "-"
    object = Object(object_number, cX, cY, p1, p2, p3, p4, angle, w, h)
    object.save_to_json(filename)


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


def shape_class(c, x, y, w, h):
    shape_estimation = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)

    # circle = cv2.HoughCircles()

    if len(shape_estimation) == 4:
        ratio = float(w) / h
        print(ratio)

        # aspect ratio to determine when a rectangle is close to be a square
        if ratio >= 0.80 and ratio <= 1.40:
            print("square")
            obj_shape = str("square")
        else:
            print("rectangle")
            obj_shape = str("rectangle")

    #

    elif len(shape_estimation) > 8:

        obj_shape = str("circle")

    return obj_shape


# uses markers to estimate the pixel to real world ratio using marker size as basis

def size_comparison(img, w, h):
    global aruco_size

    p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    corners, _, _ = aruco.detectMarkers(img, p_dict)

    # getting perimeter size of first detected marker
    perim_arcuro = cv2.arcLenght(corners[0], True)

    # size of pxl / real world size of marker
    pixel_to_real_ratio = perim_arcuro / aruco_size

    area = w * h

    ratio = float(w) / h

    if ratio >= 0.80 and ratio <= 1.20:
        obj_shape = str("square")
    else:
        obj_shape = str("rectangle")

    return obj_shape


# 3D to 2D coords transformation formula:
# https://github.com/sebastiengilbert73/tutorial_homography/blob/main/compute_homography.py
# https://towardsdatascience.com/coordinates-of-a-flat-scene-b37487df63ca


def correct_image(img):
    global mtx, dist

    h, w = img.shape[:2]
    w1, h1 = 1 * w, 1 * h

    print(h, w)

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w1, h1))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # crop the image x,y,w,h = roi dst = dst[y:y+h, x:x+w]
    x, y, w, h = roi
    print(w, h)

    dst = dst[y:y + h, x:x + w]

    return dst


# return


def _2D_3D_transf(img, valx, valy, c1, c2, c3, c4):
    global workspace_x, workspace_y, p1_1, p1_2, p2_1, p2_2, p3_1, p3_2, p4_1, p4_2, arIds, arCorner


    corners = arCorner
    ids = arIds
    print("corner count: ")
    print(len(ids))
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

    print(p1_1, p1_2, p2_1, p2_2)

    int(float(p1_1))
    int(float(p1_2))
    int(float(p2_1))
    int(float(p2_2))
    int(float(p3_1))
    int(float(p3_2))
    int(float(p4_1))
    int(float(p4_2))

    #print(p1_1, p1_2)

    # real world__________pixels
    features_mm_to_pixels_dict = {(0, 0): (p1_1, p1_2),
                                  (workspace_x, 0): (p3_1, p3_2),
                                  (workspace_x, workspace_y): (p2_1, p2_2),
                                  (0, workspace_y): (p4_1, p4_2)}

    print(features_mm_to_pixels_dict)

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

    for i in range(3):
        c1, c2, c3, c4

    print(c1, c2, c3, c4)


    val = valx
    val1 = valy

    test_xy_1 = (val, val1, 1)
    test_XY_1 = pixels_to_mm_transformation_mtx @ test_xy_1

    new_valx = test_XY_1[0]
    new_valy = test_XY_1[1]


    #print(new_valx, new_valy)

    # img_marked = aruco.drawDetectedMarkers(img, corners, ids)
    # img_marked = cv2.circle(img_marked, (val, val1), 5, (255, 125, 255), -1)
    # cv2.imshow("w", img_marked)
    # cv2.waitKey(60000)


    return new_valx, new_valy


#    mm_to_pixels_transformation_mtx = np.linalg.inv(pixels_to_mm_transformation_mtx)


def try_again():
    usr_inp = int(
        input("Would you like to try again or use this data?\n Press 1 to try again from start \n Press 2 to re-do "
              "from object addtion \n Press 3 to exit\n"))

    jsonReadTest()

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

    aruco_image_flatten(orginal_base)
    aruco_image_flatten(orginal_objects)

    if diff.getbbox():
        # diff.show()

        # convert image into array for openCV
        img4 = np.asarray(diff)

    image_gray = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)

    # cv2.imshow("mass", image_gray)

    lower = np.array([80, 81, 81])  # 79 is optimim(ish)
    higher = np.array([255, 255, 255])  # 186

    mask = cv2.inRange(image_gray, lower, higher)
    mask = cv2.blur(mask, (10, 10))
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
        print(w * h)

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

            print(h)
            print(w)

            # h  =  rect[-2]

            print(h)

            if angle < -45:
                angle = -(90 + angle)

            else:
                angle = -angle

            print(angle)

            print("''''''''''''")

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

            # change image to 3 dimensional position prediction
            realX, realY =_2D_3D_transf(img4, cX, cY, p1, p2, p3, p4)

            object = Object(object_number, realX, realY, p1, p2, p3, p4, angle, w, h)
            print("filesave...")
            # object.print_info()
            # object.save_to_json("objects.json")
            object.add_to_file("objects.json", object_number, realX, realY, p1, p2, p3, p4, angle, w, h)
            print("===================================================")

            # object counter for the session (starts at 1)
            object_number = object_number + 1

    cv2.imwrite('results.jpg', img4)
    cv2.imshow("Final Image", img4)

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

    arCorner = corners
    arIds = ids

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
    global workspace_x, workspace_y, calibrate_camera

    print("Workspace length and width are required in relation to camera POV")
    usr_inpWSx = input("Enter width(mm)  X (horizontal distance from the camera between centres of two markers)\n")
    usr_inpWSy = input("Enter length(mm) Y (Vertical distance from the camera between centres of two markers)\n")

    workspace_x = int(usr_inpWSx)
    workspace_y = int(usr_inpWSy)

    print("Width = " + str(workspace_x) + "mm")
    print("Length= " + str(workspace_y) + "mm")




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
        if calibrate_camera == 1:
             frame = correct_image(frame)
        elif calibrate_camera == 0:
            return

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

    global calibrate_camera



    cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY, params=[
        cv2.CAP_PROP_FRAME_WIDTH, 1280,
        cv2.CAP_PROP_FRAME_HEIGHT, 720])
    focus = 0  # min: 0, max: 255, increment:5
    cap.set(28, focus)

    # video capture source camera
    ret, frame = cap.read()


    # remove camera distortion

    if calibrate_camera == 1:
        frame = correct_image(frame)
    elif calibrate_camera == 0:
        return

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
