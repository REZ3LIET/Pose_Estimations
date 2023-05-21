import time
import cv2 as cv
import mediapipe as mp

class FaceTrack:
    def __init__(self, min_conf=0.5, model_type=0):
        self.mp_draw = mp.solutions.drawing_utils
        mp_face = mp.solutions.face_detection
        self.face = mp_face.FaceDetection(min_detection_confidence=min_conf, model_selection=model_type)

    def detect_face(self, img, disp=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.face.process(img_rgb)
        detected = results.detections

        if not disp:
            return detected, img
        if detected:
            for keypoints in detected:
                # print(keypoints.location_data.relative_bounding_box)
                self.mp_draw.draw_detection(img, keypoints)

        return detected, img
    
    def get_info(self, detected_keypoints, img_dims):
        bbox_info = []
        img_height, img_width = img_dims
        for id, keypoints in enumerate(detected_keypoints):
            bbox = keypoints.location_data.relative_bounding_box
            x_min, y_min, width, height = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            start_pt = int(x_min * img_width), int(y_min * img_height)
            dimensions = int(width * img_width), int(height * img_height)
            score = int(keypoints.score[0]*100)
            bbox_info.append((id, start_pt, dimensions, score))
        return bbox_info

    def draw_detection(self, bbox_info, img, conf_disp=True):
        for id, start_pt, dimensions, score in bbox_info:
            end_pt = start_pt[0] + dimensions[0], start_pt[1] + dimensions[1]
            cv.rectangle(img, start_pt, end_pt, (0, 255, 0), 10)
            if conf_disp:
                cv.putText(img, f'Score: {str(score)}', (start_pt[0], start_pt[1] - 20), cv.FONT_HERSHEY_PLAIN, 2, (0, 55, 0), 2)

        return img

def main(image=True):
    detector = FaceTrack(model_type=1)
    if image:
        ori_img = cv.imread("human_2.jpg")
        img = ori_img.copy()

        detection, output = detector.detect_face(img, disp=True)
        bounding_box_info = detector.get_info(detection, img.shape[:2])
        # print(bounding_box_info)
        output = detector.draw_detection(bounding_box_info, output)

        cv.imshow("Original", ori_img)
        cv.imshow("Detected", output)
        cv.waitKey(0)
    
    else:
        cap = cv.VideoCapture("humans_2.mp4")
        curr_time = 0
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret: 
                print("Video Over")
                break

            img = frame.copy()
            detection, output = detector.detect_face(img)
            if detection:
                bounding_box_info = detector.get_info(detection, img.shape[:2])
                print(bounding_box_info)
                output = detector.draw_detection(bounding_box_info, output)

            curr_time = time.time()
            fps = 1/(curr_time - prev_time)
            prev_time = curr_time
            cv.putText(output, f'FPS: {str(int(fps))}', (10, 70), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 50, 170), 2)

            cv.imshow("Result", output)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main(False)