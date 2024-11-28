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
target_folder1 = "cricket_players"  # Ensure this folder exists
target_folder2 = "football_players"  # Ensure this folder exists
myList = os.listdir(source_folder)
print(myList)

listImg = []

# Set up grid layout for 3 columns and 2 rows
grid_size = (3, 2)
grid_spacing = 50
start_x, start_y = 50, 50
img_index = 0

for y in range(grid_size[1]):
    for x in range(grid_size[0]):
        if img_index < len(myList):
            pathImg = myList[img_index]
            imgType = "png" if "png" in pathImg else "jpg"
            posX = start_x + x * (100 + grid_spacing)  # Adjust position based on grid
            posY = start_y + y * (100 + grid_spacing)  # Adjust position based on grid
            listImg.append(DragImg(f'{source_folder}/{pathImg}', [posX, posY], imgType))
            img_index += 1

# Load the folder images and resize them
folder_img_path1 = "folder.png"  # Replace with your folder1 image path
folder_img1 = cv2.imread(folder_img_path1, cv2.IMREAD_UNCHANGED)
folder_img1 = cv2.resize(folder_img1, (100, 100))  # Resize to 100x100 pixels

folder_img_path2 = "folder.png"  # Replace with your folder2 image path
folder_img2 = cv2.imread(folder_img_path2, cv2.IMREAD_UNCHANGED)
folder_img2 = cv2.resize(folder_img2, (100, 100))  # Resize to 100x100 pixels

# Define the drop areas for both target folders
drop_area1 = (900, 100, folder_img1.shape[1], folder_img1.shape[0])  # (x, y, width, height)
drop_area2 = (900, 300, folder_img2.shape[1], folder_img2.shape[0])  # (x, y, width, height)

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

        # Check if the image is dropped in the first drop area
        if drop_area1[0] < ox < drop_area1[0] + drop_area1[2] and drop_area1[1] < oy < drop_area1[1] + drop_area1[3]:
            # Move the file to the first target folder
            src_path = imgObject.path
            dst_path = os.path.join(target_folder1, os.path.basename(imgObject.path))
            shutil.move(src_path, dst_path)
            print(f"Moved {src_path} to {dst_path}")
            listImg.remove(imgObject)  # Remove the image from the list after moving

        # Check if the image is dropped in the second drop area
        if drop_area2[0] < ox < drop_area2[0] + drop_area2[2] and drop_area2[1] < oy < drop_area2[1] + drop_area2[3]:
            # Move the file to the second target folder
            src_path = imgObject.path
            dst_path = os.path.join(target_folder2, os.path.basename(imgObject.path))
            shutil.move(src_path, dst_path)
            print(f"Moved {src_path} to {dst_path}")
            listImg.remove(imgObject)  # Remove the image from the list after moving

    # Draw the resized folder images as the drop areas
    img = cvzone.overlayPNG(img, folder_img1, [drop_area1[0], drop_area1[1]])
    img = cvzone.overlayPNG(img, folder_img2, [drop_area2[0], drop_area2[1]])

    # Add text labels for folder names
    cv2.putText(img, target_folder1, (drop_area1[0], drop_area1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                2)
    cv2.putText(img, target_folder2, (drop_area2[0], drop_area2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                2)

    cv2.imshow("Drag_Drop", img)
    cv2.waitKey(1)



