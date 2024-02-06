import cv2
import numpy as np
import pyttsx3

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Define the class labels
class_labels = {
    0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow',
    11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
    16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
}

# Initialize the video capture object (0 for webcam, or file path for a video)
cap = cv2.VideoCapture(0)

# Initialize the text-to-speech engine
engine = pyttsx3.init()

while True:
    # Read frame from the camera
    ret, frame = cap.read()

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Set the input to the network
    net.setInput(blob)
    detections = net.forward()

    # Process the detections
    detected_objects = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Consider detections above a certain confidence threshold
            class_id = int(detections[0, 0, i, 1])

            if class_id in class_labels:
                class_name = class_labels[class_id]
                text = f'{class_name}: {confidence:.2f}%'

                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype('int')

                # Draw the prediction on the frame
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Store detected objects for audio feedback
                detected_objects.append(class_name)

    # Convert detection information to audio feedback (text-to-speech)
    if detected_objects:
        # Combine detected objects into a single string for speech output
        output_text = ', '.join(detected_objects)
        engine.say(f"I see {output_text}")
        engine.runAndWait()

    # Show the frame with detections
    cv2.imshow('Object Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
