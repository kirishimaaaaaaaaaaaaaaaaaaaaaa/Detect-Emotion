import cv2
from deepface import DeepFace

roi_window = cv2.namedWindow('ROI Select', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('ROI Select', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
face_classifier = cv2.CascadeClassifier()
face_classifier.load(cv2.samples.findFile(r"C:\Users\Lenovo\Downloads\haarcascade_frontalface_default.xml"))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(frame_gray)
    response = DeepFace.analyze(frame, actions=("emotion"), enforce_detection=False)

    for face in faces:
        x, y, w, h = face
        # print(x,y)
        newframe = cv2.rectangle(frame, (x,y), (x+w,y+h), color=(0,0,255), thickness=2)
        cv2.putText(frame, text = response[0]["dominant_emotion"], org=(x,y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=(0,0,255))
    cv2.namedWindow("emotion detection",cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("emotion detection",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.putText(frame, text = "Press Q key to exit", org=(10,470), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=(0,0,255))
    cv2.imshow("emotion detection",frame)
    if cv2.waitKey(1) == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

