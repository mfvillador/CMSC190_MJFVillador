import numpy as np
import cv2
import glob
import os


# set-up directoties for input and output
path = '../VG/raw/indx/'  
path_output = '../VG/outputs/indx/'
files = glob.glob(path + '*.jpg')   
imgcount = 0 # initialization

for file in files:     
    print(imgcount)
    print(file)
    
    original = cv2.imread(file)

    ## HSV
    image = cv2.imread(file)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # cv2.imwrite('results/hsv.jpg', hsv)

    # define range of BLUE color in HSV
    lower_blue = np.array([20,100,100])
    upper_blue = np.array([30,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    mask = cv2.dilate(mask, kernel)

    # cv2.imwrite('results/mask.jpg', mask)
    
    ## EDGING
    edged = cv2.Canny(mask, 100, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    edged = cv2.dilate(edged, kernel)
    
    ## FINDING AND EXTRACTING CNTS
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    idx = 0
    count = 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w>150 and h>150 and count<10:
            idx+=1
            new_img=original[y:y+h,x:x+w]
            cv2.imwrite(path_output + str(imgcount) + str(idx) + '.jpg', new_img)
            count = count + 1
            
    imgcount = imgcount + 1

print('DONE')