import cv2
import numpy as np
import face_recognition
import datetime
from datetime import datetime, timedelta
# inorder to add image we will now use list data structure  of python
# we will generate encodings for our images automatically

import os
path = 'Images'

myList = os.listdir(path)
print(myList)

# clear csv file
f = open("smart.csv", "w")
f.truncate()
f.close()

# list that will contain all imported images
images = []

#names of all the images without extension
classNames = []

#using this names we will import images one by one
for img in myList:
  curImage = cv2.imread(f'{path}/{img}')
  images.append(curImage)
  classNames.append(os.path.splitext(img)[0])
  
print(len(images))
print(classNames)


def attendanceMarker(name):
  with open('smart.csv','r+') as f:
    presentList=f.readlines()
    nameList=[]
    for line in presentList:
      entry=line.split(',') 
      nameList.append(entry[0])
      
    if name not in nameList:
      now=datetime.now()
      datestr=now.strftime('%H:%M:%S')
      f.writelines(f'\n{name},{datestr}')
      print(name)

 
# encoding images
def imgEncodings(images):
  encodeList = []
  for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    encodeList.append(encode)
  return encodeList

encodeList = imgEncodings(images)
print(f"Encoding complete and total of {len(encodeList)} images encoded for our use.")

# now matching images with our encoding

# webcam init
captureImage = cv2.VideoCapture(0)

# using while loop to get each frame 1 by 1
while True:
  success, img = captureImage.read()
  imgSmall = cv2.resize(img, (0,0), None, 0.25, 0.25)
  imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

  facesCurFrame = face_recognition.face_locations(imgSmall)
  encodeCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

  for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
    matches = face_recognition.compare_faces(encodeList, encodeFace)
    faceDistance = face_recognition.face_distance(encodeList, encodeFace)
    if min(faceDistance) > 0.4:
      continue
    print(faceDistance)
    matchIndex = np.argmin(faceDistance)
    
    
    if matches[matchIndex]:
      name = classNames[matchIndex].upper()
      print(name)
      attendanceMarker(name)
      #displaying content of csv
      with open('smart.csv', 'r+') as f:
        presentList = f.readlines()
        print(presentList)
      
   

      # creating bounding box and showing name of person identified
      y1, x2, y2, x1 = faceLoc
      y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
      cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
      cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

  cv2.imshow('webcam', img)
  cv2.waitKey(1)

