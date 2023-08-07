import cv2 # pip install opencv-python
import numpy as np # pip install numpy
import time
import pyttsx3 # For voice output "pip install pyttsx3"


np.random.seed(20)
class detector:
    def __init__(self,videoPath,configPath,modelPath,classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()
        self.engine = pyttsx3.init()

    def readClasses(self): # To check with each item in the class
        with open(self.classesPath,'r') as f:
            self.classesList = f.read().splitlines()

        self.colorList = np.random.uniform(low=0,high=255, size=(len(self.classesList), 3))

        

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if(cap.isOpened() == False):
            print("Error opening file...")
            return
            
        (success, image) = cap.read()

        startTime = 0

        while success:

            currentTime = time.time()
            fps = 1/(currentTime - startTime)
            startTime = currentTime
            classLabelIDs, confidences, bboxes = self.net.detect(image, confThreshold = 0.5)
            bboxes = list(bboxes)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float,confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxes, confidences, score_threshold = 0.5, nms_threshold = 0.2)

            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):

                    bbox = bboxes[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)
                    displayText1 = "{}".format(classLabel)


                    x,y,w,h = bbox

                    cv2.rectangle(image,(x,y),(x+w , y+h), color=classColor,thickness=1)
                    cv2.putText(image, displayText, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                
                    self.engine.runAndWait()
                    self.engine.say(displayText1)

                ###############################################################################
                    lineWidth = min(int(w * 0.3),int(h * 0.3))
                    cv2.line(image, (x,y),(x+lineWidth,y),classColor,thickness=5)
                    cv2.line(image, (x,y),(x, y+lineWidth),classColor,thickness=5)

                    cv2.line(image, (x+w,y),(x+w-lineWidth,y),classColor,thickness=5)
                    cv2.line(image, (x+w,y),(x+w, y+lineWidth),classColor,thickness=5)
                ###############################################################################
                    cv2.line(image, (x,y + h),(x + lineWidth,y + h),classColor,thickness=5)
                    cv2.line(image, (x,y + h),(x, y + h - lineWidth),classColor,thickness=5)

                    cv2.line(image, (x + w, y + h),(x + w - lineWidth,y + h),classColor,thickness=5)
                    cv2.line(image, (x + w,y + h),(x + w, y + h - lineWidth),classColor,thickness=5)




            cv2.putText(image, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
            cv2.imshow("Result", image)
            key = cv2.waitKey(1) &0xFF
            if key == ord("q"):
                break

                
            (success, image) = cap.read()
        cv2.destroyAllWindows()
