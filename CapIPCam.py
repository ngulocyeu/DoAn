import cv2
cap = cv2.VideoCapture('http://192.168.4.3:4747/video')
#cap.open('http://192.168.1.23:4747/video')
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    #print(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
