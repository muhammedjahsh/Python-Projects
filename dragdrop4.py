import os
import shutil
import cvzone
from cvzone.HandTrackingModule import HandDetector
import cv2

# Initialize the webcam
cam = cv2.VideoCapture(0)
cam.set(3, 1220)
cam.set(4, 850)

# Initialize the hand detector
detector = HandDetector(detectionCon=0.8)


# Define a class for draggable images
class DragImg:
    def __init__(self, path, posOrigin, imgType):
        self.posOrigin = posOrigin
        self.imgType = imgType
        self.path = path
        if self.imgType == "png":
            self.img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        else:
            self.img = cv2.imread(self.path)
        self.size = self.img.shape[:2]

    def update(self, cursor):
        ox, oy = self.posOrigin
        h, w = self.size

        if ox < cursor[0] < ox + w and oy < cursor[1] < oy + h:
            self.posOrigin = cursor[0] - w // 2, cursor[1] - h // 2


# Load images from the folder
source_folder = "face_images"
target_folder = "cricket_players"  # Ensure this folder exists
myList = os.listdir(source_folder)
print(myList)

listImg = []

for x, pathImg in enumerate(myList):
    imgType = "png" if "png" in pathImg else "jpg"
    listImg.append(DragImg(f'{source_folder}/{pathImg}', [50 + x * 300, 50], imgType))

# Load the folder image and resize it
folder_img_path = "folder.png"  # Replace with your folder image path
folder_img = cv2.imread(folder_img_path, cv2.IMREAD_UNCHANGED)
folder_img = cv2.resize(folder_img, (100, 100))  # Resize to 100x100 pixels

# Define the drop area (you can adjust the coordinates and size)
drop_area = (900, 100, folder_img.shape[1], folder_img.shape[0])  # (x, y, width, height)

# Main loop
while True:
    success, img = cam.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList']

        # Extract only the 2D coordinates (x, y) for index and middle finger tips
        index_tip = lmList[8][:2]
        middle_tip = lmList[12][:2]

        length, info, img = detector.findDistance(index_tip, middle_tip, img)
        print(length)

        if length < 60:
            cursor = index_tip
            for imgObject in listImg:
                imgObject.update(cursor)

    for imgObject in listImg:
        h, w = imgObject.size
        ox, oy = imgObject.posOrigin

        # Ensure the image does not go out of bounds
        if oy + h > img.shape[0]:
            h = img.shape[0] - oy
        if ox + w > img.shape[1]:
            w = img.shape[1] - ox

        if imgObject.imgType == "png":
            img = cvzone.overlayPNG(img, imgObject.img, [ox, oy])
        else:
            try:
                img[oy:oy + h, ox:ox + w] = imgObject.img[:h, :w]
            except ValueError as e:
                print(f"Error placing image: {e}")

        # Check if the image is dropped in the drop area
        if drop_area[0] < ox < drop_area[0] + drop_area[2] and drop_area[1] < oy < drop_area[1] + drop_area[3]:
            # Move the file to the target folder
            src_path = imgObject.path
            dst_path = os.path.join(target_folder, os.path.basename(imgObject.path))
            shutil.move(src_path, dst_path)
            print(f"Moved {src_path} to {dst_path}")
            listImg.remove(imgObject)  # Remove the image from the list after moving

    # Draw the resized folder image as the drop area
    img = cvzone.overlayPNG(img, folder_img, [drop_area[0], drop_area[1]])

    cv2.imshow("image", img)
    cv2.waitKey(1)
