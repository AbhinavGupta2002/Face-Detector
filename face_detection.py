import cv2

# Load pre-trained data on face frontals from opencv (haar cascade algorithm)
trainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Webcam
webcam = cv2.VideoCapture(0)
while True:
    successful_frame_read, frame = webcam.read()

    # Greyscale an Image
    imgGreyscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    face_coordinates = trainedFaceData.detectMultiScale(imgGreyscaled)

    # Draw figures around faces
    for face in face_coordinates:
        (x, y, w, h) = face
        # Uncomment line below and comment the "cv2.circle..."" line to display squares on each frame rather than circles
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (x + (int)(w / 2), y + (int)(h / 2)),
                   (int)(w / 2), (0, 255, 0), 3)

    cv2.imshow('Face Detector', frame)

    # Frame is refreshed every 1 millisecond
    key = cv2.waitKey(1)

    # Stop video capture when key 'q' or 'Q' is pressed
    if key == 81 or key == 113:
        break

# Release the video capture object
webcam.release()
