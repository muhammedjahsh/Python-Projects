import cv2
import pygame
import numpy as np

# Initialize pygame mixer for playing sounds
pygame.mixer.init()
pygame.mixer.music.load("vibration-rintone-13061.mp3")  # Replace with the path to your music file

# OpenCV to capture video from camera
cap = cv2.VideoCapture(0)

# Known distance from camera (in cm) for calibration (example: 50 cm)
KNOWN_DISTANCE = 20.0

# Real-world width of the object (example: 10 cm)
KNOWN_WIDTH = 5.0


# Function to compute focal length
def compute_focal_length(measured_distance, real_width, pixel_width):
    return (pixel_width * measured_distance) / real_width


# Function to compute distance to the object
def estimate_distance(real_width, focal_length, pixel_width):
    return (real_width * focal_length) / pixel_width


# Assuming you know the width of the object in pixels at the known distance (calibrate with an image)
# For example, let's say the object's width is 150 pixels when it's at 50 cm
KNOWN_PIXEL_WIDTH = 150.0

# Calculate focal length
focal_length = compute_focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, KNOWN_PIXEL_WIDTH)
print(f"Focal Length: {focal_length}")

# Set the desired distance threshold (in cm)
distance_threshold = 50.0

# Track if the music is already playing
music_playing = False

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
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Estimate the distance to the object based on its width in pixels
        distance = estimate_distance(KNOWN_WIDTH, focal_length, w)

        # Display the distance on the frame
        cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # If the object is closer than the threshold and music is not already playing, play the music
        if distance < distance_threshold and not music_playing:
            print(f"Object is within {distance_threshold} cm! Playing music...")
            pygame.mixer.music.play()
            music_playing = True
        elif distance >= distance_threshold:
            music_playing = False

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
