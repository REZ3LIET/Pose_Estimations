import time
import cv2 as cv
import mediapipe as mp

class FaceDetect:
    def __init__(self, static_image=False, max_faces=1, refine=False, detect_conf=0.5, track_conf=0.5):
        self.draw_utils = mp.solutions.drawing_utils
        self.draw_spec = self.draw_utils.DrawingSpec(color=[0, 255, 0], thickness=1, circle_radius=2)
        self.mp_face_track = mp.solutions.face_mesh
        self.face_track = self.mp_face_track.FaceMesh(static_image, max_faces, refine, detect_conf, track_conf)
    
    def detect_mesh(self, img, disp=True):
        results = self.face_track.process(img)
        detected_landmarks = results.multi_face_landmarks

        if detected_landmarks:
            if disp:
                for f_landmarks in detected_landmarks:
                    self.draw_utils.draw_landmarks(img, f_landmarks, self.mp_face_track.FACEMESH_CONTOURS, self.draw_spec, self.draw_spec)
            
        return detected_landmarks, img
    
    def get_info(self, detected_landmarks, img_dims):
        landmarks_info = []
        img_height, img_width = img_dims
        for f_id, face in enumerate(detected_landmarks):
            for id, landmarks in enumerate(face.landmark):
                x, y = int(landmarks.x * img_width), int(landmarks.y * img_height)
                landmarks_info.append((f_id, id, x, y))

        return landmarks_info
        

def main(image=True):
    detector = FaceDetect(static_image=False)
    if image:
        ori_img = cv.imread("E:\Github\Advanced_CV\Pose_Estimations\Face_Estimation\Data\Images\human_3.jpg")
        img = ori_img.copy()
        landmarks, output = detector.detect_mesh(img)
        mesh_info = detector.get_info(landmarks, img.shape[:2])
        print(mesh_info)

        cv.imshow("Result", output)
        cv.waitKey(0)

    else:
        # cap = cv.VideoCapture("E:\Github\Advanced_CV\Pose_Estimations\Face_Estimation\Data\Videos\humans_3.mp4")
        cap = cv.VideoCapture(0)
        curr_time = 0
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video Over")
                break

            img = frame.copy()

            landmarks, output = detector.detect_mesh(img)
            if landmarks:
                mesh_info = detector.get_info(landmarks, img.shape[:2])
                # print(mesh_info)

            curr_time = time.time()
            fps = 1/(curr_time - prev_time)
            prev_time = curr_time
            cv.putText(img, f'FPS: {str(int(fps))}', (10, 70), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 50, 170), 2)

            cv.imshow("Result", output)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main(False)