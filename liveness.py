import cv2

cap = cv2.VideoCapture(0)

haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    ret, frame = cap.read()
    if ret:

        faces_rects = haar_cascade_face.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
        print(' ', len(faces_rects))

        for (x, y, w, h) in faces_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            crop_image = frame[y:y + h, x:x + w]
            cv2.putText(frame,"asd",(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)

        cv2.imshow('frame',frame)
        cv2.waitKey(0)

