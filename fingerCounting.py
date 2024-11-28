from flask import Flask, render_template, Response
import cv2
import os
import time
import HandTrackingModule as hm

app = Flask(__name__)

# Setting width and height for the camera
wcam, hcam = 640, 480

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

# Path to folder containing finger images
folderpath = 'static/fingers'
myList = os.listdir(folderpath)
overlayList = []

# Loading finger images
for imgPath in myList:
    image = cv2.imread(os.path.join(folderpath, imgPath))
    overlayList.append(image)

pTime = 0

# Initialize hand detector
detector = hm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

def generate_frames():
    global pTime

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if lmList:
            fingers = []

            # Thumb
            fingers.append(1 if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] else 0)

            # Other 4 fingers
            for id in range(1, 5):
                fingers.append(1 if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2] else 0)

            totalFingers = fingers.count(1)

            if totalFingers > 0 and totalFingers <= len(overlayList):
                h, w, c = overlayList[totalFingers - 1].shape
                img[0:h, 0:w] = overlayList[totalFingers - 1]

            cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
