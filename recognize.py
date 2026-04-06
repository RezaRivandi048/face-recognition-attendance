import cv2
import pandas as pd
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

attendance_list = []

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for(x,y,w,h) in faces:
        id, conf = recognizer.predict(gray[y:y+h,x:x+w])

        if conf < 50:
            name = f"Mahasiswa {id}"
            time = datetime.now().strftime('%H:%M:%S')

            attendance_list.append([name, time])

            cv2.putText(img, name, (x,y-10), 2, 1, (0,255,0), 2)

        else:
            cv2.putText(img, "Unknown", (x,y-10), 2, 1, (0,0,255), 2)

    cv2.imshow('camera', img)

    if cv2.waitKey(10) & 0xff == 27:
        break

df = pd.DataFrame(attendance_list, columns=["Nama","Waktu"])
df.to_csv("attendance/absensi.csv", index=False)

cam.release()
cv2.destroyAllWindows()
