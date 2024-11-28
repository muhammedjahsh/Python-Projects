import cv2
import mediapipe as mp
import vlc
import threading
import tkinter as tk
import warnings
warnings.filterwarnings("ignore")

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Path to the video to display.
video_path = 'Coding_club_new.mp4'

# Function to recognize thumbs up gesture for right hand
def recognize_thumbs_up_gesture(results):
    for hand_landmarks in results.multi_hand_landmarks:
        # Check if hand is the right hand
        if hand_landmarks and hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

            # Check if thumb is up and other fingers are down
            if (thumb_tip < thumb_ip and
                index_tip > thumb_tip and
                middle_tip > thumb_tip and
                ring_tip > thumb_tip and
                pinky_tip > thumb_tip):
                return True
    return False

def play_video(video_path):
    instance = vlc.Instance()
    player = instance.media_player_new()
    media = instance.media_new(video_path)
    player.set_media(media)
    player.set_fullscreen(True)
    player.play()

    while player.get_state() != vlc.State.Ended:
        pass

    player.stop()
    instance.release()

# Flag to check if the video has been displayed.
video_displayed = False

def start_recognition():
    global video_displayed
    video_displayed = False

    # Initialize video capture for the webcam.
    cap = cv2.VideoCapture(0)
    cap.set(3, 1800)  # Set width.
    cap.set(4, 1000)  # Set height.

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Error: Could not read frame from webcam")
            break

        # Convert the image color from BGR to RGB for hand processing.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            if recognize_thumbs_up_gesture(results) and not video_displayed:
                video_displayed = True
                threading.Thread(target=play_video, args=(video_path,)).start()
                break

        # Convert the image color back to BGR for display.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Hand Gesture Recognition', image)

        # Exit on ESC key press.
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release resources and close windows.
    cap.release()
    cv2.destroyAllWindows()

# Create the main application window
root = tk.Tk()
root.title("Gesture Controlled Video Player")

# Add a button to start gesture recognition
start_button = tk.Button(root, text="Let's Reveal", command=start_recognition, width=30, height=5)
start_button.pack(pady=20)

# Run the application
root.mainloop()
