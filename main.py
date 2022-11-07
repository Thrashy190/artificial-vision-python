import cv2
import datetime

frontalface_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

video_cap = cv2.VideoCapture(0)

while True:
    data, img = video_cap.read()
    img_bn = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    smile = smile_cascade.detectMultiScale(img_bn,scaleFactor = 2,minNeighbors = 3)
    frontalface = frontalface_cascade.detectMultiScale(img_bn,scaleFactor = 1.1, minNeighbors = 5)
    eyes = eyes_cascade.detectMultiScale(img_bn,scaleFactor = 2, minNeighbors = 3)

    for fx,fy,fw,fh in frontalface:
        img = cv2.rectangle(img,(fx,fy),(fx+fw,fy+fh),(255,0,0),3)
        cv2.putText(img, 'Face', (fx, fy), cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
    
    for ex,ey,ew,eh in eyes:
        img = cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),3)
        cv2.putText(img, 'Eyes', (ex, ey), cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)


    for sx,sy,sw,sh in smile:
        img = cv2.rectangle(img,(sx,sy),(sx+sw,sy+sh),(0,0,255),3)
        cv2.putText(img, 'Smile', (sx, sy), cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

    cv2.imshow("Imagen",img)
    stop = cv2.waitKey(1)
    if stop==ord("q"):
        break

video_cap.release()
cv2.destroyAllWindows()
