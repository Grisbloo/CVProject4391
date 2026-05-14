"""
Data_CollectionV2.py
# TL;DR This program is one of the main applications user's will interact with. This collects data via turning on the user's camera showing a bounding box on their person and telling them which letter to sign.
# During the video, the landmarks are extracted from their hand position including wrist position. The landmarks are then saved as a .npy file inside a folder with the labled sign. In the case of lost data, frozen
# frames are applied so that the user can go back and re-record them with data_correction. Exit is possible via escape however no pause has yet to be implemented.
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import numpy as np
import os
import time
import sys
from frame_buffer import FrameBuffer # Import the helper for series of frames

# CONFIG
# Videos of letter in folder (Don't Change this, for simplicities sake)
num_sequences = 45 
# How many frames per video (Don't Change this, due to how LSTM's Work)
sequence_length = 16 

print("Team Members: 1=Suchi_Jain, 2=Lauren_Anderson, 3=Victor Runyan, 4=Nickolas_Ackley")
user_id = int(input("Enter your User Number (1-4): "))
action = input("Enter the letter you are recording (e.g., A): ").upper()

# The same user will only change their specific set of videos (no overriding other user's videos)
start_sequence = (user_id - 1) * num_sequences
end_sequence = user_id * num_sequences

print(f"\nUser {user_id} is recording videos {start_sequence} through {end_sequence - 1} for the letter {action}.")

# Folder location
DATA_PATH = os.path.join('ASL_Dataset')
try: 
    os.makedirs(os.path.join(DATA_PATH, action))
except OSError:
    pass

# Turns on Yolo 
yolo_model = YOLO('yolo26n.pt') 

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1, # Only use one hand for signing
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# Helper designed to exit safely without errors 
def cleanup():
    try:
        detector.close()
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()
    

# Loop through video amount (45 videos)
for sequence in range(start_sequence, end_sequence):
    # Make a fresh buffer with all zeroes to be overwritten on with no anchor point set
    window = FrameBuffer(series_length=sequence_length)
    anchor_coord = None
    last_good_keypoints = np.zeros(21 * 3)
  
    for frame_num in range(sequence_length):
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        
        yolo_results = yolo_model(frame, verbose=False)
        boxes = yolo_results[0].boxes
        
        # Default to last good known position (Frozen Frame, default Fail-safe)
        keypoints = last_good_keypoints.copy() 
        
        if len(boxes) > 0:
            x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])

            bw, bh = x2 - x1, y2 - y1
            px, py = int(bw * 0.2), int(bh * 0.2)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cropped_frame = frame[y1:y2, x1:x2]
            
            if cropped_frame.shape[0] > 0 and cropped_frame.shape[1] > 0:
                crop_h, crop_w, _ = cropped_frame.shape
                rgb_crop = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                mp_results = detector.detect(mp_image)

                if mp_results.hand_landmarks:
                    # Grab detected hand and apply landmarks to it
                    hand_landmarks = mp_results.hand_landmarks[0]
                    extracted_points = []
                    
                    # Set Anchor (Wrist)
                    wrist = hand_landmarks[0]
                    wrist_global_x = x1 + int(wrist.x * crop_w)
                    wrist_global_y = y1 + int(wrist.y * crop_h)
                    
                    # Normalize the wrist to screen scale (0 to 1)
                    wrist_norm_x = wrist_global_x / frame_w
                    wrist_norm_y = wrist_global_y / frame_h
                    wrist_z = wrist.z 

                    for landmark in hand_landmarks:
                        # Landmark in global pixels
                        global_x = x1 + int(landmark.x * crop_w)
                        global_y = y1 + int(landmark.y * crop_h)
                        
                        # Normalize all coordinates from global to screen scale (0 to 1)
                        raw_norm_x = global_x / frame_w
                        raw_norm_y = global_y / frame_h
                        
                        # Subtract the anchor
                        # This makes the wrist ALWAYS (0,0,0). 
                        # If a fingertip is at X: 0.6 and the wrist is at X: 0.5, the saved value is 0.1 (distance from wrist to tip is 0.1)
                        final_x = raw_norm_x - wrist_norm_x
                        final_y = raw_norm_y - wrist_norm_y
                        final_z = landmark.z - wrist_z 
                        
                        extracted_points.extend([final_x, final_y, final_z])
                        
                        # Visual feedback (We use the global pixels so it still draws on the user's screen properly)
                        cv2.circle(frame, (global_x, global_y), 5, (0, 255, 0), -1)
                        
                    keypoints = np.array(extracted_points)
                    last_good_keypoints = keypoints

       
        window.add_frame(keypoints)

      # User Cues
        # Draw a background bar for the status text
        cv2.rectangle(frame, (0, 0), (frame_w, 40), (0, 0, 0), -1)
        
        status_text = f"Action: {action} | Video: {sequence} | ESC to Quit"
        cv2.putText(frame, status_text, (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if frame_num == 0: 
            # Pause and allow the user to set up hand or exit in the case
            # Check for ESC 20 times in case of exit
            cv2.putText(frame, f'GET READY FOR {action}...', (120, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.imshow('Data Collection', frame)
            
            for _ in range(20): # 20 iterations * 100ms = 2 seconds
                if cv2.waitKey(100) & 0xFF == 27: 
                    print("Exiting during countdown...")
                    cleanup()
                    sys.exit()
        else: 
            # Normal operating condition
            cv2.imshow('Data Collection', frame)
            # Check for ESC during every single frame capture
            if cv2.waitKey(10) & 0xFF == 27:
                print("Exiting during capture...")
                cleanup()
                sys.exit()
            
    # Saves to file location
    # Once the loop hits 16 frames, save the 2D array as a .npy file
    npy_path = os.path.join(DATA_PATH, action, str(sequence))
    np.save(npy_path, window.get_series())

cleanup()
