
import numpy as np
import __main__

from PIL import Image, ImageChops
import numpy as np
import cv2

from objectDataSave import *


list_of = []


def try_again():
    usr_inp = int(input("Would you like to try again or use this data?\n Press 1 to try again from start \n Press 2 to re-do "
                    "from object addtion \n Press 3 to exit\n"))
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

    img1 = Image.open('image_base.jpg')
    img2 = Image.open('image_objects.jpg')

    diff = ImageChops.difference(img1, img2)

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

    #cont = sorted(cont, key=cv2.contourArea, reverse=True)

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
            print(w*h)
            if (w*h) >= 1000:
                M = cv2.moments(c)
                if M["m00"] != 0:
                  cX = int(M["m10"] / M["m00"])
                  cY = int(M["m01"] / M["m00"])
                else:
                 # set values as what you need in the situation
                 cX, cY = 0, 0

                print("object: ", i, "object centre is at: ", c[0])


                # cv2.rectangle(img4, (x, y), (x + w, y + h), (0, 255, 5), 5)
                rect = cv2.minAreaRect(c)


                box = cv2.boxPoints(rect)
                #print(box)
                box = np.int0(box)
                cv2.drawContours(img4, [box], 0, (0, 255, 0), 1)
                cv2.circle(img4, (cX, cY), 5, (255, 255, 255), -1)
                cv2.putText(img4, "object:" + str(i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

                print(box)
                print("''''''''''''")
                # print(box[0])
                # print(box[1])
                # print(box[2])
                # print(box[3])



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




                object = Object(i,cX, cY,p1, p2, p3, p4)
                print("filesave...")
                #object.print_info()
                #object.save_to_json("objects.json")
                object.add_to_file("objects.json", i, cX, cY, p1, p2, p3, p4)
                print("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW")




    cv2.imshow("Final Image", img4)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        cv2.destroyAllWindows()

    try_again()




def adding_objects():

    usr_inp6 = input("Add item to the workspace! \n Enter 1 when ready...\n (2 to exit)\n")

    if usr_inp6 == "1":
        print("Taking the Image...\n")
        # cap = cv2.VideoCapture(0)  # video capture source camera (Here webcam of laptop)
        # ret, frame = cap.read()  # return a single frame in variable `frame`
        #
        # cv2.imshow('object image', frame)
        # cv2.waitKey(6)

        usr_inp3 = input("Is the image satisfactory? y/n \n")

        if usr_inp3 == "y":
            # cv2.imwrite('image_objects.jpg', frame)
            # cv2.destroyAllWindows()
            # cap.release()

            detect_object()

        elif usr_inp3 == "n":
            adding_objects()
    elif usr_inp6 == 2:
        exit()







def take_pic():

    # cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop)
    # ret,frame = cap.read() # return a single frame in variable `frame`
    #
    # cv2.imshow('img1', frame)
    # cv2.waitKey(6)
    
    usr_inpiut2 = input("Is the image satisfactory? y/n \n")


    if usr_inpiut2 == "y":

        print("Done...")
        # cv2.imwrite('image_base.jpg', frame)
        # cv2.destroyAllWindows()
        # cap.release()

        adding_objects()



        #detect_object()




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
    usr_input = int(input("select option to begin \n Workspace layout image 1: Press 1 \n Option 2: Press 2 \n"))

    print(usr_input)

    if usr_input == 1:
        take_pic()

    elif usr_input == 2:
        print("not done")

    else:
        print("invalid input")
        image_intro()

if __name__ == '__main__':
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


