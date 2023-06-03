import cv2
import os

classNames = []
classFile = "coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres=0.45, nms=0.2):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    objectNames = []
    if len(classIds) != 0:
        for classId in classIds.flatten():
            className = classNames[classId - 1]
            objectNames.append(className)
    return objectNames

image_folder = "input/"
output_folder = "output/"

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"): # add more formats if you need
        img = cv2.imread(os.path.join(image_folder, filename))
        if img is not None:
            objects = getObjects(img)

            output_filename = os.path.splitext(filename)[0] + ".csv"
            output_filepath = os.path.join(output_folder, output_filename)

            with open(output_filepath, 'w') as f:
                for obj in objects:
                    f.write(obj + "\n")
        else:
            print(f'Unable to read image: {filename}')
