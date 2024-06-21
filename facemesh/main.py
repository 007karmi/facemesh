import cv2
import mediapipe as mp

# Initialize the MediaPipe Face Detection module
mp_face_detection = mp.solutions.face_detection.FaceDetection()
# Initialize the MediaPipe drawing utility for visualizing the detection
mp_drawing = mp.solutions.drawing_utils

# Open the default webcam
webcam = cv2.VideoCapture(0)

# Loop to continuously get frames from the webcam
while webcam.isOpened():
    success, img = webcam.read()  # Read a frame from the webcam
    if not success:
        break  # Exit the loop

    # Convert the frame color from BGRA to BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Process the frame to detect faces
    results = mp_face_detection.process(img)

    # Convert the frame color from RGB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # If faces are detected, draw the detections on the frame
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(img, detection)

    # Display the frame with detections in a window named "AMITI"
    cv2.imshow("AMITI", img)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

# Release the webcam and close any OpenCV windows
webcam.release()
cv2.destroyAllWindows()
