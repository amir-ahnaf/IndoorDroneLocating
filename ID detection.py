import cv2
import numpy as np
from imutils import grab_contours
from math import atan2

def find_rectangle(src):

    gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    gray=cv2.bilateralFilter(gray, 11, 17, 17)
    condition = 0
    dimension = gray.shape
    mask = np.zeros(dimension, np.uint8)

    edged=cv2.Canny(gray, 30, 200)
    cnts=cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts=grab_contours(cnts)
    cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt=None
    RcX, RcY = 0, 0 
    for c in cnts:
        peri=cv2.arcLength(c, True)
        approx=cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx)==4:
            screenCnt=approx
            xP1 = min(approx[0][0][0], approx[1][0][0], approx[2][0][0], approx[3][0][0],)
            yP1 = min(approx[0][0][1], approx[1][0][1], approx[2][0][1], approx[3][0][1],)
            xP2 = max(approx[0][0][0], approx[1][0][0], approx[2][0][0], approx[3][0][0],)
            yP2 = max(approx[0][0][1], approx[1][0][1], approx[2][0][1], approx[3][0][1],)
            src = src[yP1:yP2, xP1:xP2]
            condition = 1
            RcX = (xP2 - xP1)//2
            RcY = (yP2 - yP1)//2
            cv2.drawContours(src, [screenCnt], -1, (0, 255, 0), 3)
            cv2.drawContours(mask, [screenCnt], -1, (255,255,255), thickness=-1) 

    # crop quarilateral instead of rectangle
    image = cv2.bitwise_and(src, src, mask=mask)
    return src, condition, RcX, RcY


def findCode(pattern):

    i,j,k = 0, 0, 0

    pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

    # divide into zones
    # cv2.shape[0] = height,    cv2.shape[1] = width
    TriZone = pattern[pattern.shape[1]//3 : pattern.shape[1]*2//3, 0: pattern.shape[0]//3]
    Zone1 = pattern[ : pattern.shape[1]//2, pattern.shape[1]//3 : pattern.shape[1]*2//3]
    Zone2 = pattern[pattern.shape[1]//2 : , pattern.shape[1]//3 : pattern.shape[1]*2//3]
    Zone3 = pattern[ : pattern.shape[1]//2, pattern.shape[1]*2//3 : ]
    Zone4 = pattern[pattern.shape[1]//2 : , pattern.shape[1]*2//3 : ]

    Zones = [TriZone, Zone1, Zone2, Zone3, Zone4]
    code = ''
    cX, cY = 0,0

    for zone in Zones:
        ret,thresh = cv2.threshold(zone,127,2,1)
        contours,h = cv2.findContours(thresh,1,2)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1]
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            if len(approx)==3:
                cv2.drawContours(pattern,[cnt],0,(0,255,0),1)
                code += '3'
                M = cv2.moments(cnt)
                M = list(M.items())
                cX = int(M[1][1] / M[0][1])
                cY = int(M[2][1] / M[0][1])

            elif len(approx)==4:
                cv2.drawContours(pattern,[cnt],0,(0,0,255),1)
                code += '1'
            elif len(approx) > 4:
                cv2.drawContours(pattern,[cnt],0,(0,255,255),1)
                code += '0'
    cv2.imshow('rectangle', pattern)

    return code, cX, cY

# def findDetails():


capture = cv2.VideoCapture(0)
capture1 = cv2.VideoCapture(2)
capture2 = cv2.VideoCapture(3)

if not capture.isOpened():
    print("Cannot open cam")
    exit()

while True:
    isTrue, frame = capture.read()
    isTrue, frame1 = capture1.read()
    isTrue, frame2 = capture2.read()

    frame = cv2.flip(frame, -1)
    condition = 0
    frame, condition, RcX, RcY = find_rectangle(frame)
    frame1, condition, RcX1, RcY1 = find_rectangle(frame1)
    frame2, condition, RcX2, RcY2 = find_rectangle(frame2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,200,1)
    contours,h = cv2.findContours(thresh,1,2)

    if (condition):
        Code, TcX, TcY = findCode(frame)

        if (Code[0] != '0') and (Code[0] !='1')  :
            
            if Code == '0001':
                drone = 1
            
            elif Code == '0010':
                drone = 2

            elif Code == '0011':
                drone = 3

            elif Code == '0100':
                drone = 4

            elif Code == '0101':
                drone = 5

            elif Code == '0110':
                drone = 6
            
            elif Code == '0111':
                drone = 7

            elif Code == '1000':
                drone = 8

            elif Code == '1001':
                drone = 9

            elif Code == '1010':
                drone = 10

            print(f'Drone number {drone},   Coordinate: ({RcX},{RcY})')

            # degree
            x1 = RcX - TcX
            y1 = RcY - TcY
            yow = atan2(x1, y1)
            yow = yow * 180 / 3.142
            print(f'Yow: {yow} degree')


    cv2.imshow('Live Streaming', frame)
    cv2.imshow('Live Streaming1', frame1)
    cv2.imshow('Live Streaming2', frame2)

    
    if not isTrue:
        print("Cant retrieve frame")
        break
    

    cv2.imshow('Live Streaming',frame)
    if cv2.waitKey(1) == ord('q'):
        break


