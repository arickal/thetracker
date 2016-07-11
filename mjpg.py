import cv2
import numpy as np
import urllib
import time

current_milli_time = lambda: int(round(time.time() * 1000))

stream=urllib.urlopen('http://arduino.local:8080/?action=stream/frame.mjpg')
bytes=''
t = -1
lower_red = np.array([0,100,100])
upper_red = np.array([30,255,255])
kernel = np.ones((5,5),np.uint8)

while True:
    bytes+=stream.read(1024)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')


    if a!=-1 and b!=-1:
        jpg = bytes[a:b+2]
        bytes= bytes[b+2:]
        img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_red, upper_red)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        ret,thresh = cv2.threshold(mask,127,255,0)
        res = cv2.bitwise_and(img,img, mask= mask)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        if len(contours)>0:
            cnt = contours[0]
            x,y,w,h = cv2.boundingRect(cnt)
            if w>50 and h>50:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                
        cv2.imshow('img',img)
        if cv2.waitKey(1) ==27:
            exit(0) 
