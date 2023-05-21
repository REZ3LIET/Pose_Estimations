import time
import cv2 as cv
import numpy as np
import mediapipe as mp

class HandPoseDetect:
    def __init__(self, static_image=False, max_hands=2, complexity=1, detect_conf=0.5, track_conf=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image, max_hands, complexity, detect_conf, track_conf)

    def detect_landmarks(self, img, disp=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        detected_landmarks = results.multi_hand_landmarks

        if detected_landmarks:
            if disp:
                for h_landmark in detected_landmarks:
                    self.mp_draw.draw_landmarks(img, h_landmark, self.mp_hands.HAND_CONNECTIONS)
        return detected_landmarks, img

    def get_info(self, detected_landmarks, hand_no, img):
        lm_list = []
        if not detected_landmarks:
            return lm_list

        if hand_no > 2:
            print('[WARNING] Provided hand number is greater than max number 2')
            print('[WARNING] Calculating information for hand 2')
            hand_no = 2
        elif hand_no < 1:
            print('[WARNING] Provided hand number is less than min number 1')
            print('[WARNING] Calculating information for hand 1')

        if len(detected_landmarks) < 2:
            hand_no = 0
        else:
            hand_no -= 1

        height, width, _ = img.shape
        for id, h_landmarks in enumerate(detected_landmarks[hand_no].landmark):
            cord_x, cord_y = int(h_landmarks.x * width), int(h_landmarks.y * height)
            lm_list.append([id, cord_x, cord_y])

        return lm_list

def main(image=False):
    detector = HandPoseDetect()
    if image:
        ori_img = cv.imread("typing.jpg")
        cv.imshow("Original", ori_img)

        img = ori_img.copy()
        landmarks, output_img = detector.detect_landmarks(img)
        info_landmarks = detector.get_info(landmarks, 3, img)
        print(info_landmarks)

        cv.imshow("Landmarks", output_img)
        cv.waitKey(0)

    else:
        cap = cv.VideoCapture("piano_playing.mp4")
        prev_time = time.time()
        cur_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video Over")
                break

            img = frame.copy()
            landmarks, output_img = detector.detect_landmarks(img)
            info_landmarks = detector.get_info(landmarks, 3, img)

            cur_time = time.time()
            fps = 1/(cur_time - prev_time)
            prev_time = cur_time
            cv.putText(output_img, f'FPS: {str(int(fps))}', (10, 70), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 50, 170), 2)

            cv.imshow("Detection", output_img)
            cv.imshow("Original", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    cv.destroyAllWindows()


if __name__ == "__main__":
    main(image=False)
