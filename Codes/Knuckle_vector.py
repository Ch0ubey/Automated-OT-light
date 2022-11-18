from tkinter import Image
from turtle import left, right
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
c2,c1,a1,a2,b1,b2,d1,d2 = 0,0,0,0,0,0,0,0
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)
mpDraw = mp.solutions.drawing_utils

def detectHandsLandmarks(image, hands, display = True):
    
    output_image = image.copy()
    
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        
        for hand_landmarks in results.multi_hand_landmarks:
            
            mpDraw.draw_landmarks(image = output_image, landmark_list = hand_landmarks,
                                  connections = mpHands.HAND_CONNECTIONS) 
    
    if display:
        
        plt.figure(figsize=[15,15])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output");plt.axis('off');
        
    else:
        
        return output_image, results

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hand_no, handLms in enumerate(results.multi_hand_landmarks):
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                output_image = img.copy()
                hands_status = {'Right': False, 'Left': False, 'Right_index' : None, 'Left_index': None}
                for hand_index, hand_info in enumerate(results.multi_handedness):
        
                    hand_type = hand_info.classification[0].label
                          
                    hands_status[hand_type] = True
                    hands_status[hand_type + '_index'] = hand_index

                    if hand_no==0:
                        print("move")

                    if (id == 9):
                        # print('move')
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                        # print(hand_type, hand_no, id, cx, cy)
                        if hand_type == "Right":
                             a1=cx 
                             b1=cy
                             
                        if hand_type == "Left":
                             a2=cx 
                             b2=cy
                        
                    if id == 10:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                        if hand_type == "Right":
                             c1=cx 
                             d1=cy
                        if hand_type == "Left":
                             c2=cx 
                             d2=cy
                        # print(hand_type, hand_no, id, cx, cy)
                        # cv2.arrowedLine(img, start_point, end_point,color, thickness)
                    if ((pow(((d1-b1)*(d1-b1)+(c1-a1)*(c1-a1)),0.5)) !=0) & ((pow(((c2-a2)*(c2-a2)+(d2-b2)*(d2-b2)),0.5)) != 0): 
                        fxd= (c1-a1)/(pow(((d1-b1)*(d1-b1)+(c1-a1)*(c1-a1)),0.5)) + (c2-a2)/(pow(((c2-a2)*(c2-a2)+(d2-b2)*(d2-b2)),0.5))
                        fyd= (d1-b1)/(pow(((d1-b1)*(d1-b1)+(c1-a1)*(c1-a1)),0.5)) + (d2-b2)/(pow(((c2-a2)*(c2-a2)+(d2-b2)*(d2-b2)),0.5))
                        # if hand_no==0:
                        #  hand_type=="Right"
                        #  cv2.arrowedLine(img, (a1,b1), (c1,d1),(0,255,0), 3)
                        # if hand_no==1:
                        #  hand_type == "Left"
                        #  cv2.arrowedLine(img, (a2,b2), (c2,d2),(255,0,0), 3)
                          
                        print(fxd,' i    ' ,fyd,' j' )
                        # start_point = (((a1+a2)/2), ((b1+b2)/2))
                        # end_point = (((c1+c2)/2), ((d1+d2)/2))
                        # color = (0, 255, 0)
                        # thickness = 2
                        # print(start_point)
                        # print(end_point)
                        # cv2.arrowedLine(img, start_point, end_point,color, thickness)
                    
                    
    cv2.imshow("Image", img)
    cv2.waitKey(1)