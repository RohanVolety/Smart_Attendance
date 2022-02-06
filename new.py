import cv2
import os
import numpy as np
import face_recognition


imgBill = face_recognition.load_image_file('dataset/face3.jpeg')
imgBill = cv2.cvtColor(imgBill, cv2.COLOR_BGR2RGB)

imgBill1 = face_recognition.load_image_file('dataset/face5.jpeg')
imgBill1= cv2.cvtColor(imgBill1, cv2.COLOR_BGR2RGB)




faceloc = face_recognition.face_locations(imgBill)[0]
encodeBill = face_recognition.face_encodings(imgBill)[0]
cv2.rectangle(imgBill, (faceloc[3], faceloc[0]),
              (faceloc[1], faceloc[2]), (0, 255, 0), 2)


faceloc1 = face_recognition.face_locations(imgBill1)[0]
encodeBill1 = face_recognition.face_encodings(imgBill1)[0]
cv2.rectangle(imgBill1, (faceloc1[3], faceloc1[0]),
              (faceloc1[1], faceloc1[2]), (0, 255, 0), 2)


result=face_recognition.compare_faces([encodeBill],encodeBill1)
face_distance=face_recognition.face_distance([encodeBill],encodeBill1)

cv2.putText(imgBill1,f'{result}{round(face_distance[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

print(result)
print(face_distance)

path = 'dataset'
mylist=os.listdir(path)

print(mylist)


cv2.imshow("image",imgBill)
cv2.imshow("image1", imgBill1)
cv2.waitKey(0)
