import time
import cv2 as cv
import mediapipe as mp

class BodyPoseDetect:
    def __init__(self, static_image=False, complexity=1, smooth_lm=True, segmentation=False, smooth_sm=True, detect_conf=0.5, track_conf=0.5):
        self.mp_body = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.body = self.mp_body.Pose(static_image, complexity, smooth_lm, segmentation, smooth_sm, detect_conf, track_conf)

    def detect_landmarks(self, img, disp=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.body.process(img_rgb)
        detected_landmarks = results.pose_landmarks

        if detected_landmarks:
            if disp:
                self.mp_draw.draw_landmarks(img, detected_landmarks, self.mp_body.POSE_CONNECTIONS)
        return detected_landmarks, img
    
    def get_info(self, detected_landmarks, img):
        lm_list = []
        if not detected_landmarks:
            return lm_list

        height, width, _ = img.shape
        for id, b_landmark in enumerate(detected_landmarks.landmark):
            cord_x, cord_y = int(b_landmark.x * width), int(b_landmark.y * height)
            lm_list.append([id, cord_x, cord_y])
        
        return lm_list

def main(image):
    detector = BodyPoseDetect()
    if image:
        ori_img = cv.imread("Pose_Estimations\\Body_Estimation\\Data\\Images\\dance.jpg")
        
        img = ori_img.copy()
        landmarks, output_img = detector.detect_landmarks(img)
        info_landmarks = detector.get_info(landmarks, img)
        print(info_landmarks[3])

        cv.imshow("Original", ori_img)
        cv.imshow("Detection", output_img)
        cv.waitKey(0)
    
    else:
        cap = cv.VideoCapture("Pose_Estimations\\Body_Estimation\\Data\\Videos\\pushup.mp4")
        prev_time = time.time()
        cur_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video Over")
                break

            img = frame.copy()
            landmarks, output_img = detector.detect_landmarks(img)
            info_landmarks = detector.get_info(landmarks, img)
            # print(info_landmarks[3])

            cur_time = time.time()
            fps = 1/(cur_time - prev_time)
            prev_time = cur_time
            cv.putText(output_img, f'FPS: {str(int(fps))}', (10, 70), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 50, 170), 2)

            cv.imshow("Original", frame)
            cv.imshow("Detection", output_img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
    
    cv.destroyAllWindows()

if __name__ == "__main__":
    main(image=False)