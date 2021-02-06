import cv2

#loading the classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

#capturing the video input
video = cv2.VideoCapture(0)
while True:
    check,frame = video.read()
    #getting frames

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detecting front_face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    #drawing rectangle around the faces
    for x, y, w, h in faces:
        image = cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0), 2)
        image[y:y+h,x:x+w] =cv2.medianBlur(image[y:y+h,x:x+w],35)
    #creating GUI for video
    cv2.imshow("Blurred Face", frame)

    #conditional variable to close the app
    key = cv2.waitKey(20)
    if key==27:             #27 is the code for 'Esc' button
        break

video.release()
cv2.destroyAllWindows

