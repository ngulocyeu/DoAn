#ObjectTracking.py
import random
import threading
import numpy as np
import cv2 as cv
import time
import math
import imutils
#import numba as nb
from datetime import datetime
from urllib.request import urlopen
from imutils.object_detection import non_max_suppression
#from control import *
from imutils import paths
from convert_video import convert_video
#from detected_image import detected_image
import matplotlib.pyplot as plt
import json
from tkinter import *
from paho.mqtt import client as mqtt_client
import itertools
from distanceCalc import distanceCalc

#define threading wrappe
frameWidth = 640
frameHeight = 192

green = (0,255,0)
red = (0,0,255)
blue = (255,0,0)
purple = (205,50,219)

maxSpeed = 50
maxAngle = 45
minAngle = -45

broker = 'mqtt.ngoinhaiot.com'
port = 1111
topic1 = "huuminh1999hp/quat"
topic2 = "huuminh1999hp/321"
topic3 = "huuminh1999hp/maylanh"
# generate client ID with pub prefix randomly
#client_id = f'python-mqtt-{random.randint(0, 1000)}'
client_id = '246'
username = 'huuminh1999hp'
password = 'huuminh1999hp'

#fourcc = cv.VideoWriter_fourcc(*'AVID')
#out = cv.VideoWriter('new.mp4', fourcc, 20.0, (640,360))

tracking = False
displayStream = False
showClock = False
showFPS = True
nms = True
record = False
displayDetect = False
is_on = False
vitri = None
Chon = None

    
def Publish(client,imageFrame):
        
    depthFrame = convert_video(imageFrame)
    depthFrame = cv.cvtColor(depthFrame, cv.COLOR_BGR2GRAY)
                
            #Detect People
    boundingBoxes,labels = detectPeople(imageFrame)
            #print(boundingBoxes)  
            #Add Bounding Boxes
    frame = addBoundingBoxes(imageFrame,boundingBoxes,green)
                
            #Add the Goal Position
    frame, goalPosition = addGoal(frame,purple)

    if len(boundingBoxes) > 0:

        print("INFO = Phat Hien "+str(len(boundingBoxes))+" Doi Tuong.")
                
        frame = addText(frame,"Tracking Active",red)

        ObjectCenters = selectObject(boundingBoxes)

        frame = addMarker(frame,ObjectCenters,green)

        if displayDetect == True:
                
            cv.imshow('Detecting {}',frame)

        def on_message(client, userdata, message):
                            global Chon
                            time.sleep(1)
                            print("received message =",str(message.payload.decode("utf-8")))
                            Chon = message.payload.decode("utf-8")
                            return Chon
                    
        client.on_message=on_message
               

        Labels = []

        Chons = []

        for i in range(len(labels)):

            Label = str(labels[i])+str(i+1)

            Labels.append(Label)

        print(Labels)

        Chiso = []

        ChuoiSo = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n']

        for i in range(len(labels)):

            chiso = str(ChuoiSo[i])

            Chiso.append(chiso)
                    
        #print(Chiso)

        Ds = [val for pair in zip(Chiso,Labels) for val in pair]

        #print(Ds)

        def Convert(lst):
                res_dct = {lst[i]:lst[i + 1] for i in range(0, len(lst), 2)}
                return res_dct

        DS = Convert(Ds)

        #print(DS)
        DanhSach = json.dumps(DS)

        #print(DanhSach)
                
        msg1 = DanhSach
        send1 = client.publish(topic1, msg1)
        status1 = send1[0]
                    
        if status1 == 0:
                    
                            print(f"Send `{msg1}` to topic `{topic1}`")
        else:
                    
                            print(f"Failed to send message to topic {topic1}")

        get = client.subscribe(topic3)
        status2 = get[0]

        if status2 == 0:
                    
                            print(f"get message from topic `{topic3}`")
        else:
                    
                            print(f"Failed to get message from topic {topic3}")

        Chons.append(Chon)
                #print(Chons)
                
        for i in range(len(Labels)):

                if (Chons[0] == Labels[i]):

                        vitri = i

                        print('OK')

                        break
                    
                else:

                        vitri = None

                        continue
                        
                #print(vitri)
                           
        if vitri == None:
                    
                    print("Hay Chon Doi Tuong")
                    
        else:
                    
                    boundingBox = boundingBoxes[vitri]

                    #select Object

                    ObjectPosition = ObjectCenters[vitri]

                    #Determine Image Size
                    width = frame.shape[1] 
                    height = frame.shape[0]

                    #print(ObjectCenters)             
                    #Chose object
                    #boundingBox = button(labels)

                    #print(boundingBox)
                    #Add Crosshair Markers                
                   # print(ObjectCentres)
                    #In an image with multiple people select a Object to follow
                        
                    #Distance calc
                    distance = calcObjectDistance(ObjectPosition,depthFrame)
                    angle = calcAngle(goalPosition,ObjectPosition,height,width)

                    output = {'0': str(distance),'1': str(angle)}

                    OutPut = json.dumps(output)

                    msg3 = OutPut
                    send2 = client.publish(topic2, msg3)
                    status3 = send2[0]
                    
                    if status3 == 0:
                            print(f"Send `{msg3}` to topic `{topic2}`")
                    else:
                            print(f"Failed to send message to topic {topic2}")
                   
                    #Add Crosshair and Bounding Box for Target Object
                    frame = addMarker(frame,[ObjectPosition],red)
                    frame = addBoundingBoxes(frame,[boundingBox],red)
                          
                    text1 = "Distance: "+str(round(distance, 2))
                    text2 = "Angle: "+str(angle)

                    font = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(frame,text1,(16,68), font, 0.6,(0,0,255),1,cv.LINE_AA)
                    cv.putText(frame,text2,(16,88), font, 0.6,(0,0,255),1,cv.LINE_AA)
                    for i in range(len(boundingBoxes)):
                        x, y, a, b = boundingBoxes[i]
                        label = labels[i]
                        #confidence = confidences[i]    
                    
                        cv.putText(frame, label ,(x, y+20),font, 0.5,(0,0,255),2)

    else:

                #Distance = calcFontDistance(depthFrame)
                
                a = {'0': '0.3', '1': '20'}
                
                Output = json.dumps(a)

                msg1 = Output
                send1 = client.publish(topic1, msg1)
                status1 = send1[0]
                    
                if status1 == 0:
                    
                        print(f"Send `{msg1}` to topic `{topic3}`")
                else:
                        print(f"Failed to send message to topic {topic3}")
                
                #wheelchair.transmitCommand(0,0,"RUN")
                frame = addText(frame,"Khong Co Vat Can",green)
                text = "Xoay Xe Lan"
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(frame,text,(16,68), font, 0.6,(0,255,0),1,cv.LINE_AA)
                
           # print(distance)


            #if record == True:
                    #out.write(frame)
            #print(distance)
            #KhoangCach = json.dumps(distance)
            #DoLech = json.dumps(offset)
                
            #Calculate FPS
                
    #time.sleep(adjustedDelay)

    cv.waitKey(10)

    return frame
    

#Collision Prevention
    
def getFrames(image):

    #returned, depthFrame  = depth.read()
    returned, imageFrame  = image.read()

    if returned == False:
        print("ERROR = Cannot Access Vision API.")
        #depthFrame = cv.imread('nostream.jpg',cv.IMREAD_COLOR)
        #imageFrame = cv.imread('nostream.jpg',cv.IMREAD_COLOR)
        nostream = True
        #Convert Depth Image to Grayscale
        #depthFrame = cv.cvtColor(depthFrame, cv.COLOR_BGR2GRAY)
    else:
        nostream = False
        imageFrame = imutils.resize(imageFrame, width=frameWidth)
    #depthFrame = imutils.resize(depthFrame, width=frameWidth)

    return imageFrame, nostream
            

def addClock(frame):

    #Add clock to the frame
    font = cv.FONT_HERSHEY_SIMPLEX
    currentDateTime = datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")
    cv.putText(frame,currentDateTime,(16,20), font, 0.6,(255,0,0),1,cv.LINE_AA)

    return frame
    

def addFPS(frame,fps):

    #Add clock to the frame
    font = cv.FONT_HERSHEY_SIMPLEX
    text = '%.2ffps'%round(fps,2)
    cv.putText(frame,text,(16,44), font, 0.6,(255,0,0),1,cv.LINE_AA)

    return frame


def addText(frame,text,colour):
    
    font = cv.FONT_HERSHEY_SIMPLEX            
    cv.putText(frame,text,(16,110), font, 0.6,colour,1,cv.LINE_AA)

    return frame

def detectPeople(image):
    
    net = cv.dnn.readNet('yolov3.weights','cfg\yolov3.cfg')
    classes = []
    with open ('Data\coco.names','r') as f:
                classes = f.read().splitlines()
    height, width, _ = image.shape
    blob = cv.dnn.blobFromImage(image,1/255,(416,416),(0,0,0), swapRB = True, crop = False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    Boxes = []
    confices = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confice = scores [class_id]
            if confice > 0.5:
                center_x = int(detection[0]* width)
                center_y = int (detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                Boxes.append([x,y,w,h])
                confices.append((float(confice)))
                class_ids.append(class_id)
    
    if len(Boxes) > 0:
        
        #(boundingBoxes, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.2)
        #Boxes,Distances,confidences = detected_image(image)      
        indexes = cv.dnn.NMSBoxes(Boxes, confices, 0.5, 0.4)
        #font = cv.FONT_HERSHEY_PLAIN
        #colors = np.random.uniform(0, 255, size=(len(Boxes), 3for i in indexes.flatten():
        #confidences = []
        boundingBoxes = []
        labels = []
        for i in indexes.flatten():
            x, y, w, h = Boxes[i]
            label = str(classes[class_ids[i]])
            #confidence = str(round(confices[i],2))

            boundingBoxes.append([x, y, x+w, y+h])
            #confidences.append(confidence)
            labels.append(label)
    else:
        
        boundingBoxes = ()
        labels = ()
        
    return boundingBoxes,labels

def selectObject(boundingBoxes):

  if  len(boundingBoxes) > 0:
        i = 0
        ObjectCenters = []
        for (xA, yA, xB, yB) in boundingBoxes: 
            x = int(((xB -xA)/2) + xA)
            y = int(((yB -yA)/2) + yA)
            ObjectCenters.insert(i,(x,y))
            i = i + 1
  else:
      
     ObjectCenters = 0

  return ObjectCenters


def addMarker(image,points,colour):
        
    crosshairHeight = 10
    crosshairWidth = 10

    for (x, y) in points:
        #Horizontal Line & Vertical Line on Video Image
        cv.line(image,((x-crosshairWidth),y),((x+crosshairWidth),y),colour,2)
        cv.line(image,(x,(y-crosshairHeight)),(x,(y+crosshairHeight)),colour,2) 

    return image


def addBoundingBoxes(image,boxes,colour):
        
    #Draw boxes without NMS
    for (xA, yA, xB, yB) in boxes:
        cv.rectangle(image, (xA, yA), (xB, yB),colour, 2)
  
    return image


def addGoal(image,colour):
        
    offset = 0
    crosshairHeight = 10
    crosshairWidth = 10

    width = image.shape[1] 
    height = image.shape[0]

    goalWidth = int((width/2) - offset)
    goalHeight = int((height/2) - offset)
        
    goalPosition = [goalHeight, goalWidth]

    #Horizontal Line & Vertical Line on Video Image
    cv.line(image,((goalWidth-crosshairWidth),goalHeight),((goalWidth+crosshairWidth),goalHeight),colour,2)
    cv.line(image,(goalWidth,(goalHeight-crosshairHeight)),(goalWidth,(goalHeight+crosshairHeight)),colour,2)

    return image, goalPosition


    
def calcAngle(goalPosition,ObjectPosition,height,width):
    
    maxAngle = 100
    minAngle = -100

    xG = goalPosition[0]
    xP = ObjectPosition[0]

    mappingRange = width/2

    if xP > xG:
            offset = maxAngle * (xP - xG)/mappingRange

    elif xP < xG:
            offset = minAngle * (xG - xP)/mappingRange

    else:
            offset = 0

    return offset   
     
    #Providing the Location of a Object, returns their distance away
def calcObjectDistance(ObjectPosition,depthFrame):
        
    [y,x] = ObjectPosition
    
    depthFrame = cv.medianBlur(depthFrame,5)
    depthValue1 = depthFrame[x-10,y-10]
    depthValue2 = depthFrame[x-10,y+10]
    depthValue3 = depthFrame[x+10,y-10]
    depthValue4 = depthFrame[x+10,y+10]
    #depthValue5 = depthFrame[x+10,y]
    #depthValue6 = depthFrame[x,y+10]
    #depthValue7 = depthFrame[x-10,y]
    #depthValue8 = depthFrame[x,y-10]
    depthValue = round((int(depthValue1)+int(depthValue2)+int(depthValue3)+int(depthValue4))/4)
    distance = distanceCalc(depthValue)
    
    if distance <= 0.4:
        distance = 0.4
    else:
        if distance >= 4.6:
            distance = 4.6
                
    return distance

#Returns infomraiton about how far away a point is in and image
#@staticmethod
#@staticmethod
def scanImage(depthData):
        
    height = len(depthData)
    width = len(depthData[0])

    #Initialise with worst case
    pointValue = 2048
    pointHeight = 0
    pointWidth = 0

        #Threshold for dealing with annomolies (reflective surfaces)
    threshold = 0

        #opulate Array with Data
    for h in range (0,height):

       for w in range (0,width):

               if (depthData[h,w] <= pointValue) and (depthData[h,w] >= threshold):
                   pointValue = depthData[h,w]
                   pointHeight = h
                   pointWidth = w
                
                   results = [pointValue, pointWidth, pointHeight]
        
       return results
    
def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client
