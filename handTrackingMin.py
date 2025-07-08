import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands #formality before writing a module
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


pTime = 0
cTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  #for multiple hands
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) #convert co-ordinates into pxl
                print(id,cx,cy) #print id of location of all 20 points
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255,0,255, cv2.FILLED))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  #to draw connections and dots

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN, 3,
                (225,0,225),3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)