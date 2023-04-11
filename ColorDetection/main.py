import cv2
import numpy as np
from collections import deque

# deque size for center of object
buffer_size = 16
pts = deque(maxlen=buffer_size)

# lower boundary RED color range values; Hue (0 - 10)
lower1 = np.array([0, 100, 20])
upper1 = np.array([10, 255, 255])
 
# upper boundary RED color range values; Hue (160 - 180)
lower2 = np.array([160,100,20])
upper2 = np.array([179,255,255])

#get capture
cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 480)

while True:
    success , img = cap.read()
    if success:
        #blured image 
        blurred = cv2.GaussianBlur(img, (11,11) , 0)

        # transform HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # check hsv format
        # cv2.imshow("HSV image",hsv)

        # mask for red
        lower_mask = cv2.inRange(hsv, lower1, upper1)
        upper_mask = cv2.inRange(hsv, lower2, upper2)
        full_mask = lower_mask + upper_mask
        # check mask 
        # cv2.imshow("Mask image",full_mask)

        # delete noise after mask
        mask = cv2.erode(full_mask,None,iterations=2)
        mask = cv2.dilate(mask,None,iterations=2)
        # check delete erode
        # cv2.imshow("Deleted erode",mask)

        #find contours
        (contours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(contours) > 0:
            # get max contour
            c = max(contours, key=cv2.contourArea)

            # convert to rect
            rect = cv2.minAreaRect(c)

            ((x,y), (w,h), rotation) = rect

            # box
            box = np.int64(cv2.boxPoints(rect))

            # moment for center
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
            pts.appendleft(center)
            
            # draw contour
            cv2.drawContours(img,[box],0,(255,255,255),2)

            # point of center
            cv2.circle(img,center,5,(255,0,255),-1)

        for i in range(1,len(pts)):
            if pts[i-1] is None or pts[i] is None:
                continue
            cv2.line(img, pts[i-1], pts[i], (0,255,0), 3)
            
        cv2.imshow("Last Version of color detection",img)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break