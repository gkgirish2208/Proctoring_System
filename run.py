'''
import cv2
import dlib
import time

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load pre-trained face and eye detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Dlib face detector and facial landmarks
# Make sure the 'shape_predictor_68_face_landmarks.dat' is in the same directory or provide the correct path
predictor_path = "G:\sony\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Helper function to determine if eyes are closed
def eyes_closed(eyes):
    for (ex, ey, ew, eh) in eyes:
        if eh / ew > 0.3:
            return False
    return True

# Initialize state trackers
last_blink_time = time.time()
last_attention_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract the face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Draw rectangles around eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Check if eyes are closed
        if eyes_closed(eyes):
            current_time = time.time()
            if current_time - last_blink_time > 1:  # Adjust time threshold as needed
                label = "Sleepy"
                last_blink_time = current_time
            else:
                label = "Attentive"
        else:
            label = "Attentive"
            last_attention_time = time.time()

        # Check if face is looking away
        if len(eyes) < 3:
            current_time = time.time()
            if current_time - last_attention_time > 2:  # Adjust time threshold as needed
                label = "Non-Attentive"
        
        # Display the label
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Student State Detector', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
'''


import cv2
import dlib
import time

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load pre-trained face and eye detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Dlib face detector and facial landmarks
# Make sure the 'shape_predictor_68_face_landmarks.dat' is in the same directory or provide the correct path
predictor_path = "G:\sony\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Helper function to determine if eyes are closed
def eyes_closed(eyes):
    for (ex, ey, ew, eh) in eyes:
        if eh / ew > 0.3:
            return False
    return True

# Helper function to determine if hand is close to face
def hand_close_to_face(face_landmarks, hand_point):
    # Check if hand point is close to any face landmarks
    for point in face_landmarks.parts():
        distance = abs(hand_point[0] - point.x) + abs(hand_point[1] - point.y)
        if distance < 20:  # Adjust this threshold as needed
            return True
    return False

# Initialize state trackers
last_blink_time = time.time()
last_attention_time = time.time()

# Dictionary to store last label for each face
last_label_dict = {}

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract the face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Draw rectangles around eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Get facial landmarks
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        face_landmarks = predictor(gray, dlib_rect)
        
        # Check if hand is close to face (simulated by checking if any face landmark is close to a predefined point)
        hand_point = (x + w // 2, y + h + 30)  # Assuming the hand is below the face
        if hand_close_to_face(face_landmarks, hand_point):
            label = "Using Phone"
        else:
            # Check the number of visible eyes
            num_eyes = len(eyes)
            # Initialize label as Attentive
            label = "Attentive"
            if num_eyes == 0:
                # Both eyes closed
                current_time = time.time()
                if current_time - last_blink_time > 0.3:  # Threshold for eyes closed
                    label = "Sleepy"
                    last_blink_time = current_time
            elif num_eyes == 1:
                # One eye closed
                label = "Non-Attentive"
            else:
                # Both eyes open
                last_attention_time = time.time()

            # Check if face is looking away
            if len(eyes) < 2:
                current_time = time.time()
                if current_time - last_attention_time > 2:  # Threshold for attention
                    label = "Bored"
        
        # Display the label only if it has changed or is not yet assigned
        if (x, y) not in last_label_dict or last_label_dict[(x, y)] != label:
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            last_label_dict[(x, y)] = label
    
    # Display the resulting frame
    cv2.imshow('Student State Detector', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
