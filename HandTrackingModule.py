import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands  # formality before writing a module
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:  # for multiple hands
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),  # pink dots
                        self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)   # green lines
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # convert co-ordinates into pxl
                # print(id, cx, cy)  # print id of location of all 20 points
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 6, (255, 0, 0), cv2.FILLED)  # pink dot with 6px radius
        return lmList

def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)  # fixed from 1 to 0 for default webcam
    detector = handDetector()
    while True:
        success, img = cap.read()
        if not success:
            continue  # skip frame if capture failed

        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
