from Detect import *
import os

def main():
    videoPath = 0  # For video try "test_video/video.mp4"

    configPath = os.path.join("model_data","ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data","frozen_inference_graph.pb")
    classesPath = os.path.join("model_data","coco.names")

    detect = detector(videoPath,configPath,modelPath,classesPath)
    detect.onVideo()

if __name__ == '__main__':
    main()