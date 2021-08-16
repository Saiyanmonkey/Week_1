import cv2 as cv

face_detector = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
smile_detector = cv.CascadeClassifier('haarcascade_smile.xml')

cap = cv.VideoCapture(0)

while True:
    _, img = cap.read()
    
    img = cv.resize(img,(0,0),fx=0.6,fy=0.6)
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    
    #deteCT FACES
    
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minSize=(50,50),
        minNeighbors=5,
        )
    
    for (x,y,w,h) in faces:
        
        cv.rectangle(img, (x, y), ((x + w), (y + h)), (0, 0, 255), 2)
        face = img[y:y+h,x:x+w]
        gray_face = gray[y:y+h,x:x+w]
        
        #detecting smiles
        smiles = smile_detector.detectMultiScale(
            gray_face,
            scaleFactor=2,
            minSize=(20,40),
            minNeighbors=8,
            
            )
        
        for (sx,sy,sw,sh) in smiles:
            cv.rectangle(img, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)
            cv.rectangle(face, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
            
    
    cv.imshow('img',img)
    if cv.waitKey(30) & 0xff == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()