import cv2

face_cascade = cv2.CascadeClassifier(r"output/classifier/test/haarcascades/haarcascade_frontalface_default.xml")

capture = cv2.VideoCapture(0)

while capture.isOpened():
    _, img = capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 15)
    if len(faces) > 0:
        print(faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.imshow('detect_img', img)
        break
    cv2.imshow('detect_img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
