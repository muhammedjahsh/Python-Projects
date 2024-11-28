import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize video capture, face mesh detector, and get screen dimensions
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Variables to track blink timing for double click detection
right_blink_time = 0
right_blink_count = 0

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmarkspoint = output.multi_face_landmarks
    frame_h, frame_w = frame.shape[0], frame.shape[1]

    if landmarkspoint:
        landmarks = landmarkspoint[0].landmark

        # Eye tracking and cursor movement
        for id, land in enumerate(landmarks[474:478]):
            x = int(land.x * frame_w)
            y = int(land.y * frame_h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0))
            if id == 1:
                screen_x = screen_w / frame_w * x
                screen_y = screen_h / frame_h * y
                pyautogui.moveTo(screen_x, screen_y)

        # Blink detection for left eye (single click)
        left = [landmarks[145], landmarks[159]]
        for land in left:
            x = int(land.x * frame_w)
            y = int(land.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

        if (left[0].y - left[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(2)

        # Blink detection for right eye (double click)
        right = [landmarks[374], landmarks[386]]
        for land in right:
            x = int(land.x * frame_w)
            y = int(land.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 0, 255))

        # Check for a single blink
        if (right[0].y - right[1].y) < 0.004:
            current_time = time.time()
            if current_time - right_blink_time < 0.5:  # Check if the blink is within 0.5 seconds of the last one
                right_blink_count += 1
                if right_blink_count == 2:
                    pyautogui.doubleClick()  # Perform double click
                    right_blink_count = 0
            else:
                right_blink_count = 1
            right_blink_time = current_time

    cv2.imshow("Eye Controlling", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()