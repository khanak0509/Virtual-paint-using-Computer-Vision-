import cv2 
import mediapipe as mp
import numpy as np

import time 
canvas = np.zeros((720, 1280, 3), np.uint8)

ptime = 0 
ctime = 0 
capture = cv2.VideoCapture(0)
mphand = mp.solutions.hands
hands = mphand.Hands()
mpdraw = mp.solutions.drawing_utils
header = cv2.imread("color.png")
capture.set(3, 1280)
thickness = 8
capture.set(4, 720)
xp , yp = None,None
draw_color = (255, 0, 255)

while True:
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    success , img = capture.read()
    if not success:
        break

    img = cv2.flip(img,1)
    img[0:100, 0:1280] = header
    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    lm_list = []
    result = hands.process(imgrgb)
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            for id , lm in enumerate(handlms.landmark):
                h,w,c = img.shape
                cx , cy = int(lm.x*w),int(lm.y*h)

                lm_list.append([id,cx,cy])

    
    if lm_list:
        x1 = lm_list[8][1]
        y1 = lm_list[8][2]
        x2 = lm_list[12][1]
        y2 = lm_list[12][2]

        if lm_list[8][2]<lm_list[7][2] and lm_list[12][2]<lm_list[11][2] :
                
                xp, yp = None, None 
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)

             #selection
                if 0 < y1 < 100:
                        if 0 < x1 < 256:
                            thickness = 20
                            draw_color = (255, 0, 255)
                        elif 256 < x1 < 512:
                            thickness = 20
                            draw_color = (255, 255, 0)
                        elif 512 < x1 < 768:
                            thickness = 20
                            
                            draw_color = (255, 0, 0)
                        elif 768 < x1 < 1024:
                            draw_color = (0, 0, 0)
                            thickness = 50
                        elif 1024 < x1 < 1280:
                            thickness = 20
                            draw_color = (255, 255, 255)
                        

        elif lm_list[8][2] < lm_list[7][2] and lm_list[12][2] > lm_list[11][2]:
                if xp is None and yp is None:
                        xp, yp = x1, y1
                if draw_color == (0, 0, 0):  # eraser mode
                    cv2.line(canvas, (xp, yp), (x1, y1), (0, 0, 0), thickness)
                    cv2.line(img, (xp, yp), (x1, y1), draw_color, thickness)
                else:
                    cv2.line(canvas, (xp, yp), (x1, y1), draw_color, thickness)
                    cv2.line(canvas, (xp, yp), (x1, y1), draw_color, thickness)

                xp =x1
                yp = y1
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)
        
              
    


    cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()