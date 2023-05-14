import os
import wget

MODELS = {
    "lightning.tflite": {"url": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite", 
                  "loc": "..\\Models\\lightning.tflite"},
    "thunder.tflite": {"url": "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite", 
                  "loc": "..\\Models\\thunder.tflite"},
    "multipose.tflite": {"url": "https://tfhub.dev/google/movenet/multipose/lightning/1?tf-hub-format=compressed", 
                  "loc": "..\\Models\\multipose.tflite"}
}

def model_exist(model_name):
    if model_name in os.listdir("..\\Models"):
        print(f"{model_name} found!!!")
        return True
    print(f"{model_name} not found!!!")
    return False

def download_model(model_name):
    if not model_exist(model_name):
        url = MODELS[model_name]["url"]
        output = MODELS[model_name]["loc"]
        wget.download(url, out=output)
        print("Downloaded Model")