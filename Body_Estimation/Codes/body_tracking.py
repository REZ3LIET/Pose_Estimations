import os
import cv2 as cv
import numpy as np
import tensorflow as tf

from model_downloader import download_model

class BodyTracker:
    EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
    }
    def __init__(self, model_name):
        # Preparing Model
        # Checking and downloading model
        if model_name not in os.listdir("..\\Models"):
            download_model(model_name=model_name)

        # Loading Model
        self.interpreter = tf.lite.Interpreter(model_path=f"..\Models\\{model_name}")
        self.interpreter.allocate_tensors()
    
    def get_keypoints(self, image):
        # Setup input and output
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Processing image for tensor input
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_with_pad(image, 192, 192)
        input_img = tf.cast(image, dtype=tf.uint8)
        self.interpreter.set_tensor(input_details[0]['index'], np.array(input_img))
        self.interpreter.invoke()
        keypoints_with_score = self.interpreter.get_tensor(output_details[0]['index'])
        return keypoints_with_score

    def draw_keypoints_edges(self, image, keypoints, confidence_thresh):
        height, width, _ = image.shape
        shaped = np.squeeze(np.multiply(keypoints, [height, width, 1]))
        for kp in shaped:
            kp_y, kp_x, kp_conf = kp
            if kp_conf >= confidence_thresh:
                cv.circle(image, (int(kp_x), int(kp_y)), 5, (0, 225, 0), -1)
        
        for edge, _ in self.EDGES.items():
            kp_1, kp_2 = edge
            y_1, x_1, c_1 = shaped[kp_1]
            y_2, x_2, c_2 = shaped[kp_2]

            if c_1 >= confidence_thresh and c_2 >= confidence_thresh:
                cv.line(image, (int(x_1), int(y_1)), (int(x_2), int(y_2)), (0, 0, 225), 2)

def main(path, model_name="lightning.tflite"):
    tracker = BodyTracker(model_name=model_name)
    # Differentiate path between image vid and camera
    # Dependng on read from source
    img = cv.imread(path)
    img = cv.resize(img, (640, 480))
    keypoints_with_score = tracker.get_keypoints(image=img.copy())
    tracker.draw_keypoints_edges(img, keypoints_with_score, 0.02)
    cv.imshow("Result", img)
    cv.waitKey(0)
    cv.destroyAllWindows()



if __name__ == "__main__":
    path = "E:\Github\Advanced_CV\Pose_Estimations\Body_Estimation\Data\Images\humans_1.jpg"
    main(path)