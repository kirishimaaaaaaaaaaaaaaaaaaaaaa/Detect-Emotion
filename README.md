Emotion Detection using DeepFace & OpenCV
Overview

This project captures real-time video from your webcam, detects faces using OpenCV’s Haar Cascade classifier, and predicts the dominant emotion on each detected face using the DeepFace library. The detected emotions are displayed as text above the bounding boxes around faces.

Features

Real-time face detection

Emotion recognition: Happy, Sad, Angry, Surprise, Neutral, etc.

Fullscreen window for video feed

Simple UI with instructions to exit (Press Q key to exit)

Requirements

Python 3.8+

OpenCV

DeepFace

NumPy

Install dependencies using pip:

pip install opencv-python opencv-python-headless deepface numpy

Code Explanation

Imports:

import cv2
from deepface import DeepFace


cv2 is OpenCV for image/video processing.

DeepFace performs emotion analysis.

Face Detection Setup:

face_classifier = cv2.CascadeClassifier()
face_classifier.load(cv2.samples.findFile("path_to_haarcascade_frontalface_default.xml"))


Loads Haar Cascade for frontal face detection.

Make sure the .xml file path is correct.

Video Capture:

cap = cv2.VideoCapture(0)


Captures video from the default webcam.

Main Loop:

ret, frame = cap.read()
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(frame_gray)
response = DeepFace.analyze(frame, actions=("emotion"), enforce_detection=False)


ret, frame = cap.read()

ret: Boolean indicating if the frame was captured successfully.

frame: The actual video frame.

detectMultiScale detects faces in the grayscale image.

DeepFace.analyze predicts emotions. enforce_detection=False avoids crashing if no face is detected.

Drawing Bounding Boxes and Labels:

for face in faces:
    x, y, w, h = face
    cv2.rectangle(frame, (x,y), (x+w,y+h), color=(0,0,255), thickness=2)
    cv2.putText(frame, response[0]["dominant_emotion"], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))


Draws a rectangle around detected faces.

Displays the dominant emotion above the rectangle.

Display Window:

cv2.imshow("emotion detection", frame)


Shows the video feed with overlays.

Press Q to exit.

Cleanup:

cap.release()
cv2.destroyAllWindows()


Releases the webcam and closes all OpenCV windows.

Mistakes to Avoid

Cascade Classifier Loading:

Always provide the correct path for .xml file.

Using cv2.samples.findFile() is safer than hardcoding paths.

Window Creation:

Use cv2.namedWindow() once before the loop.

Avoid setting fullscreen repeatedly inside the loop.

DeepFace Analysis:

Running DeepFace.analyze for every frame can be slow. Consider analyzing every few frames to improve performance.

Error Handling:

Always check ret before processing the frame:

if not ret:
    print("Failed to capture frame")
    continue


Code Cleanup:

Avoid creating windows or setting properties multiple times inside the loop.

Example Usage
python emotion_detection.py


A fullscreen window will open showing your webcam feed.

Detected faces will have rectangles with the dominant emotion label.

Press Q to quit.

If you want, I can also rewrite your code properly, fixing all the mistakes and making it faster and cleaner so it won’t lag with DeepFace.

Do you want me to do that?

You said:
use emojies and make it fun and a little shorter
ChatGPT said:

Here’s a fun, emoji-filled, and slightly shorter README version for your project:

😄 Real-Time Emotion Detection 🎥
What it does

This project uses your webcam to:

Detect faces 🫵

Recognize emotions 😃😢😡😮😐

Show the emotion above each face in real-time 🖼️

Requirements

Python 3.8+ 🐍

OpenCV

DeepFace

NumPy

Install with:

pip install opencv-python opencv-python-headless deepface numpy

How It Works

1️⃣ Face Detection

face_classifier = cv2.CascadeClassifier()
face_classifier.load("haarcascade_frontalface_default.xml")


Detects faces using OpenCV’s Haar Cascade.

2️⃣ Webcam Capture

ret, frame = cap.read()


ret: ✅ True if frame captured successfully

frame: 🎞️ The actual video frame

3️⃣ Emotion Recognition

response = DeepFace.analyze(frame, actions=("emotion"), enforce_detection=False)


Predicts emotions like Happy, Sad, Angry, Surprise, Neutral.

4️⃣ Draw & Display

cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
cv2.putText(frame, response[0]["dominant_emotion"], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
cv2.imshow("emotion detection", frame)


Shows rectangles around faces with the detected emotion 💌

5️⃣ Exit

Press Q to quit ❌

⚠️ Tips & Common Mistakes

✅ Always check ret before using the frame.

✅ Load the .xml file correctly.

⚡ Avoid running DeepFace on every frame—can be slow.

🖼️ Don’t create windows inside the loop every frame.

Run it
python emotion_detection.py


Watch your emotions get detected in real-time! 😎🎉
