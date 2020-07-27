# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:23:45 2020

@author: ACER
"""


# import the necessary packages
from utils import social_distancing_config as config
from utils.detection import detect_people
from itertools import combinations
import math
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
inputFile="example.mp4"
opFile="output.avi"

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3-tiny.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3-tiny.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
cam = cv2.VideoCapture(inputFile if inputFile else 0)
writer = None
fps=cam.get(5)
print(fps)
# loop over the frames from the video stream
frameCount=0
while True:
    # read the next frame from the file
    (ret, frame) = cam.read()
    
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not ret:
        break
    
    # resize the frame and then detect people (and only people) in it
    frame = imutils.resize(frame, width=800)
    h,w=frame.shape[:2]
    # if frameCount%5==0:
    results = detect_people(frame, net, ln,personIdx=LABELS.index("person"))
    red_zone_list = []
    for res1,res2 in combinations(results, 2):
            dx, dy = res1[2][0] - res2[2][0], res1[2][1] - res2[2][1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < 50.0:
                if res1 not in red_zone_list:
                    red_zone_list.append(res1)
                if res2 not in red_zone_list:
                    red_zone_list.append(res2)
                    
    for res in results:
        conf=res[0]
        x1,y1,x2,y2=res[1]
        cx,cy=res[2]
        if res in red_zone_list:
            cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 0, 255), 2)
            cv2.circle(frame,(cx,cy),3,(0,0,255),2,1)
        else:
            cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)
            cv2.circle(frame,(cx,cy),3,(0,255,0),2,1)
    # for res in results:
    #     conf=res[0]
    #     x1,y1,x2,y2=res[1]
    #     cx,cy=res[2]
        
    #     cv2.rectangle(frame,(x1,y1-10),(x2,y2+10),(255,0,0),2,1)
    #     cv2.circle(frame,(cx,cy),3,(255,0,0),2,1)
    
    cv2.putText(frame,"Social Distancing Violations:"+str(len(red_zone_list)),(10,h-10),2,0.7,(0,0,255),2,1)
    # print(results)
    
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==27:
        break
    
    frameCount+=1

    # if an output video file path has been supplied and the video
    # writer has not been initialized, do so now
    if opFile != "" and writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(opFile, fourcc, 25,
            (frame.shape[1], frame.shape[0]), True)

    # if the video writer is not None, write the frame to the output
    # video file
    if writer is not None:
        writer.write(frame)
cam.release()
cv2.destroyAllWindows()
    