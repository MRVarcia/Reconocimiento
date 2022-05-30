import os

import cv2

data_path = f'data'
image_paths = os.listdir(data_path)
print(f'image_paths = {image_paths}')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.read('modeloLBPHFace.xml')

cap = cv2.VideoCapture(0)

face_classif= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascades_frontalface_defaul.xml')

while True:
    ret, frame = cap.read()
    if not  ret : break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_frame= gray.copy()

    faces= face_classif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = aux_frame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(frame, f'{result}', (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        if result[1] < 70:
            cv2.putText(frame, f'{image_paths[result[0]]}', (x, y - 25),2, 1.1, (0, 255, 0),1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 0, 255),2 )

    cv2.imshow('frame', frame)
    k = cv2.waitkey(1)
    if k == 27
        break

cap.release()
cv2.destroyAllWindows()


