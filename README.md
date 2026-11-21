# Workshop-2---Object-detection-using-web-camera
## NAME : JANANI D
## REG NO : 212225040142


## Aim
To access the laptop camera, capture an image, and detect objects using YOLOv8.

## Requirements
Anaconda
Jupyter Notebook
Laptop Camera
Steps
Open Jupyter Notebook using Anaconda.
Access your laptop camera using OpenCV.
Wait for 5 seconds and capture an image.
Display the captured image.
Apply YOLOv8 object detection.
Display the detected image with bounding boxes and labels.
## Algorithm
Laptop Camera
      ↓
Capture Image
      ↓
Display Image
      ↓
YOLOv8 Detection
      ↓
Display Detected Objects
Student Task
Capture your own image using the laptop camera.
Detect the objects present in the image.
Display the final output.
Perform the experiment with 3 different scenes.
Submission
Jupyter Notebook (.ipynb)
Original captured images
YOLOv8 output images
Screenshot of the final result
code
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from IPython.display import display,clear_output
import time
from PIL import Image

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

print("Camera started...")
print("Capturing frame in 5 seconds...")

# Wait for 5 seconds
time.sleep(10)

# Capture a single frame
ret, frame = cap.read()

if ret:
    # Perform object detection
    results = model(frame)

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    img=cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    # Display the detected frame
    #clear_output(wait=True)
    #plt.figure(figsize=(8,6))
    #plt.imshow(img)
    #plt.axis("off")
    #plt.title("YOLOv8 Object Detection")
    #plt.show()
    display(Image.fromarray(img))
else:
    print("Failed to capture frame.")

# Release webcam
cap.release()


## OUTPUT:

<img width="911" height="672" alt="Screenshot 2026-11-21 113819" src="https://github.com/user-attachments/assets/4179eff8-79fd-4199-9f46-ec08f4142b88" />

