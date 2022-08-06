
from PIL import Image, ImageChops
import cv2
import numpy as np

img1 = Image.open('empty.jpg')
img2 = Image.open('group.jpg')

diff = ImageChops.difference(img1, img2)

if diff.getbbox():
   #diff.show()

   #convert image into array for openCV
   img4 = np.asarray(diff)

image_gray = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)

#cv2.imshow("mass", image_gray)


lower = np.array([79,79,79]) # 79 is optimim(ish)
higher = np.array([186,186,186]) # 186

mask = cv2.inRange(image_gray, lower, higher)
mask = cv2.blur(mask,(10,10))
cv2.imshow("mask",mask)


cont,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)

cont = sorted(cont, key=cv2.contourArea, reverse=True)

#cont = sorted(cont, key=lambda x:cv2.boundingRect(x)[0])

new = cv2.drawContours(image_gray,cont, -1,(0,255,0),2);
cv2.imshow( "2",new)
print("number of cnt", len(cont))



for (i,c) in enumerate(cont):
    # alternative method that draws around contours
    #epsilon = 0.01* cv2.arcLength(cnt,True)
    #approx = cv2.approxPolyDP(cnt, epsilon, True)
    #img4 = cv2.drawContours(img4, [approx], 0, (0, 255, 0), 2)

    #mx = max(cont, key=cv2.contourArea)

    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        # set values as what you need in the situation
        cX, cY = 0, 0

    print(c[[0]])


         #  x, y, h, w = cv2.boundingRect(mx)
        #cv2.rectangle(img4, (x, y), (x + w, y + h), (0, 255, 5), 5)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)



    cv2.drawContours(img4, [box], 0, (0, 255, 0), 1)
    cv2.circle(img4, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(img4,"object:" + str(i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)


cv2.imshow( "0", img4)

cv2.waitKey(0)