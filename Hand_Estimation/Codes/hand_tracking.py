import time
import cv2 as cv
import numpy as np
import mediapipe as mp

class HandTracker:
    def __init__(self, mode=False, maxHands=2, detectionConf=0.5, trackConf=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=mode,
                                        max_num_hands=maxHands,
                                        model_complexity=1,
                                        min_detection_confidence=detectionConf, 
                                        min_tracking_confidence=trackConf)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, blank=True, draw=True):
        if blank:
            out_img = np.zeros(img.shape, dtype=np.uint8)
        else:
            out_img = img.copy()

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(out_img, handLms, self.mp_hands.HAND_CONNECTIONS)
        return out_img

def main(path=None):
    tracker = HandTracker()

    if path is None:
        cap = cv.VideoCapture(0)  # For webcam source
    else:    
        cap = cv.VideoCapture(path)  # For video source

    prev_time = time.time()
    cur_time = 0

    while True:
        ret, img = cap.read()
        blank = np.zeros(img.shape, dtype=np.uint8)

        if not ret:
            break

        res = tracker.find_hands(img)

        if path is None:
            output = cv.flip(res, 1)
            sleep = 1
        else:
            output = res
            sleep = 20

        cur_time = time.time()
        fps = 1/(cur_time - prev_time)
        prev_time = cur_time
        cv.putText(output, f'FPS: {str(int(fps))}', (10, 70), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 50, 170), 2)

        cv.imshow("Frames", output)
        cv.imshow("Original Input", img)
        if cv.waitKey(sleep) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    path = 'E:\Github\Advanced_CV\Pose_Estimations\Hand_Estimation\Data\piano_playing.mp4'
    main(path)
