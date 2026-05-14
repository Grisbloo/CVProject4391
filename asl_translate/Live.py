# TL;DR This program is the final showcasing and putting the model on display to use. It uses the path created in Train_Model.py to create a yolo26n.pt to use as the model it runs on. There is a HUD that displays
# the confidence and the letter being signed. There is a yolo bounding box to hold in the person, mediapipe hands to provide the landmarking, and the LSTM to provide the classification with its own created path.
# It also shows the user the landmarks of their hand. All the letters are available to sign.

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from frame_buffer import FrameBuffer
from ultralytics import YOLO
import numpy as np
import torch
import threading
import time
import sys
from ASL_Model import ASLSequenceInterpreter

# Config
# Available actions match training data 
actions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
buffer = FrameBuffer(series_length=16) # Holds rolling window of frames

# Minimum confidence to display a prediction. Below this shows "..." instead.
CONFIDENCE_THRESHOLD = 0.7

# Only re-run YOLO every N frames. Reuses last box in between for speed.
YOLO_SKIP_FRAMES = 8
sequence_length = 16


# THREADED CAMERA READER
# Same pattern as Data_collection2.py — reads frames in background so the main loop never blocks waiting on the camera hardware.
class CameraReader:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def cleanup(cam):
    """MediaPipe first, then camera, then windows. Order matters."""
    try:
        detector.close()
    except Exception:
        pass
    cam.release()
    cv2.destroyAllWindows()


# Load Models (Yolo and Mediapipe)
print("Loading YOLO...")
yolo_model = YOLO('yolo26n.pt')

print("Loading MediaPipe...")
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

print("Loading LSTM brain...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Running on: {device}")
model = ASLSequenceInterpreter(num_classes=len(actions)).to(device)
model.load_state_dict(torch.load('best_asl_model.pth', map_location=device))
model.eval()


# Variables
sequence = []                        # Rolling 16-frame window
current_prediction = "..."           # Guess of the letter
current_confidence = 0.0             # Softmax confidence of that prediction
last_box = None                      # Persisted YOLO box
yolo_counter = 0                     # Frame counter for YOLO throttle
last_good_keypoints = np.zeros(21 * 3) # Failsafe (zeroes instead of a frozen frame since we can return to ... rather than a incorrect letter)


# Start Camera
print("Starting camera...")
cam = CameraReader(0)

for _ in range(30):  # Wait up 1.5s for first frame
    if cam.read() is not None:
        break
    time.sleep(0.05)

print("We are doing it live! Press ESC to quit.")

# Main Loop 
while True:
    frame = cam.read()
    if frame is None:
        continue

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    # Throttled version of Yolo to improve frame rate
    if yolo_counter % YOLO_SKIP_FRAMES == 0:
        yolo_results = yolo_model(frame, imgsz=320, verbose=False)
        boxes = yolo_results[0].boxes
        if len(boxes) > 0:
            x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
            # 20% padding to the bounding box to maintain hand in box
            bw, bh = x2 - x1, y2 - y1
            px, py = int(bw * 0.2), int(bh * 0.2)
            last_box = (
                max(0, x1 - px), max(0, y1 - py),
                min(frame_w, x2 + px), min(frame_h, y2 + py)
            )
        else:
            last_box = None
    yolo_counter += 1

    # MEDIAPIPE: every frame using persisted box to keep accuracy high
    keypoints = last_good_keypoints.copy()

    if last_box is not None:
        x1, y1, x2, y2 = last_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cropped = frame[y1:y2, x1:x2]

        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            crop_h, crop_w, _ = cropped.shape
            rgb_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
            mp_results = detector.detect(mp_image)

            if mp_results.hand_landmarks:
                # Grab detected hand and apply landmarks to it
                hand_landmarks = mp_results.hand_landmarks[0]
                extracted_points = []
                
                # Set the Anchor
                wrist = hand_landmarks[0]
                # Wrist in global pixel coordinates
                wrist_global_x = x1 + int(wrist.x * crop_w)
                wrist_global_y = y1 + int(wrist.y * crop_h)
                
                # Normalize the wrist to screen scale (0 to 1)
                wrist_norm_x = wrist_global_x / frame_w
                wrist_norm_y = wrist_global_y / frame_h
                wrist_z = wrist.z 

                for landmark in hand_landmarks:
                    # Turn landmarks into global
                    global_x = x1 + int(landmark.x * crop_w)
                    global_y = y1 + int(landmark.y * crop_h)
                    
                    # Normalize screen scale (0 to 1)
                    raw_norm_x = global_x / frame_w
                    raw_norm_y = global_y / frame_h
                    
                    # Subtract the Anchor
                    # This makes the wrist ALWAYS (0,0,0). 
                    final_x = raw_norm_x - wrist_norm_x
                    final_y = raw_norm_y - wrist_norm_y
                    final_z = landmark.z - wrist_z 
                    
                    extracted_points.extend([final_x, final_y, final_z])
                    
                    # Visual feedback (We use the global pixels so it still draws on the screen properly)
                    cv2.circle(frame, (global_x, global_y), 5, (0, 255, 0), -1)
                    
                keypoints = np.array(extracted_points)
                last_good_keypoints = keypoints # Update the failsafe memory

    # Rolling Window
    # Always use zeros when no hand found — honest signal to the LSTM (Allows ... to exist)
    sequence.append(keypoints)
    sequence = sequence[-sequence_length:]

    # LSTM Prediction
    if len(sequence) == sequence_length:
        input_data = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            raw_output = model(input_data)

            # Softmax turns raw scores into probabilities that sum to 1
            probs = torch.softmax(raw_output, dim=1)
            confidence, predicted_index = torch.max(probs, dim=1)

            current_confidence = confidence.item()
            if current_confidence >= CONFIDENCE_THRESHOLD:
                current_prediction = actions[predicted_index.item()]
            else:
                # Model isn't sure enough — don't guess
                current_prediction = "..."

    # HUD containing bounding box, confidence, letter, if there is hand or not
    cv2.rectangle(frame, (0, 0), (frame_w, 65), (0, 0, 0), -1)

    hand_status = "Hand: YES" if last_box is not None else "Hand: NO"
    conf_pct = f"{current_confidence * 100:.0f}%"
    status_line = f"{hand_status}  |  Confidence: {conf_pct}  |  ESC to Quit"
    cv2.putText(frame, status_line, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    pred_color = (0, 255, 0) if current_prediction != "..." else (100, 100, 100)
    cv2.putText(frame, f"Prediction: {current_prediction}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, pred_color, 3, cv2.LINE_AA)

    cv2.imshow('ASL Live Translation', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("Exiting...")
        break

cleanup(cam)
sys.exit()
