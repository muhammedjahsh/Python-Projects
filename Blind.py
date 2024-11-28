import cv2
import pygame
import numpy as np

# Initialize pygame mixer for playing sounds
pygame.mixer.init()
pygame.mixer.music.load("vibration-rintone-13061.mp3")  # Load your tune

# OpenCV to capture video from camera
cap = cv2.VideoCapture(0)

# Set the threshold for object size (adjust based on your scenario)
size_threshold = 3000

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale and blur to reduce noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges in the frame
    edged = cv2.Canny(blur, 50, 150)

    # Find contours in the edged frame
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours
    for contour in contours:
        # Get the area of the contour
        area = cv2.contourArea(contour)

        # If the object area exceeds the threshold, play the tune
        if area > size_threshold:
            print("Object is close! Playing tune...")
            pygame.mixer.music.play()

            # Draw the contour and bounding box on the frame
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
