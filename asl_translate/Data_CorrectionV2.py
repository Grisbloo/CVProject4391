import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import numpy as np
import os
import sys
import time
from frame_buffer import FrameBuffer

# CONFIG (DO NOT CHANGE THIS)
user_input = input("Enter letters to check and fix (e.g., A B C) or ALL: ").upper()
DATA_ROOT = 'ASL_Dataset'
sequence_length = 16 

# Parse the input into a list
if user_input == "ALL":
    actions = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
else:
    actions = user_input.replace(',', ' ').split()

# Check for file path
for action in actions:
    DATA_PATH = os.path.join(DATA_ROOT, action)
    
    if not os.path.exists(DATA_PATH):
        print(f"[!] Directory {DATA_PATH} does not exist. Skipping...")
        continue
        
    print(f"\n{'='*40}")
    print(f"=== NOW PROCESSING LETTER: {action} ===")
    print(f"{'='*40}")

    # Check for unusable data
    print(f"Scanning {DATA_PATH} for frozen frame drops...")
    bad_sequences = []
    
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".npy"):
            filepath = os.path.join(DATA_PATH, filename)
            try:
                data = np.load(filepath)
                
                # Check 1: All zeroes?
                if np.all(data == 0):
                    seq_num = int(filename.replace('.npy', ''))
                    bad_sequences.append(seq_num)
                    print(f"[{filename}] CRITICAL FAIL: File is completely empty (Pure Zeros).")
                    continue
                    
                # Check 2: Max amount of Frozen Frames allowed
                FROZEN_THRESHOLD = 5
                frozen_count = 0
                
                for i in range(1, data.shape[0]):
                    # If the current frame is EXACTLY equal to the previous frame it is frozen
                    if np.array_equal(data[i], data[i-1]):
                        frozen_count += 1
                        
                if frozen_count >= FROZEN_THRESHOLD:
                    seq_num = int(filename.replace('.npy', ''))
                    bad_sequences.append(seq_num)
                    print(f"[{filename}] FAIL: Too many drops. {frozen_count}/16 frames were frozen.")
                    
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    if not bad_sequences:
        print("-" * 40)
        print(f"Success! No drops or frozen frames found in the {action} dataset.")
        continue
    
    bad_sequences.sort()
    print("-" * 40)
    print(f"Corrupted/Frozen sequences identified: {bad_sequences}")
    print(f"Total files to re-record: {len(bad_sequences)}")
    input("Press ENTER to start the camera and patch these files...")
    
    # Re-Recording Setup
    yolo_model = YOLO('yolo26n.pt') 
    
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1, 
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(0)
    
    def cleanup():
        try:
            detector.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
    
    # Only overwrite the bad sequences
    for sequence in bad_sequences:
        window = FrameBuffer(series_length=sequence_length)
        
        # Anchor cordinate to match with the data collection anchor coord
        anchor_coord = None
        last_good_keypoints = np.zeros(21 * 3)
        
        for frame_num in range(sequence_length):
            success, frame = cap.read()
            frame = cv2.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape
            
            yolo_results = yolo_model(frame, verbose=False)
            boxes = yolo_results[0].boxes
            
            # Default to the failsafe (filling array last known good keypoints (frozen))
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
                        
                        # Frame Anchor (The Wrist)
                        wrist = hand_landmarks[0]
                        # We need the wrist in global pixel coordinates first
                        wrist_global_x = x1 + int(wrist.x * crop_w)
                        wrist_global_y = y1 + int(wrist.y * crop_h)
                        
                        # Normalize the wrist to screen scale (0 to 1)
                        wrist_norm_x = wrist_global_x / frame_w
                        wrist_norm_y = wrist_global_y / frame_h
                        wrist_z = wrist.z 
    
                        for landmark in hand_landmarks:
                            # Global landmarking
                            global_x = x1 + int(landmark.x * crop_w)
                            global_y = y1 + int(landmark.y * crop_h)
                            
                            # Normalize screen scale (0 to 1)
                            raw_norm_x = global_x / frame_w
                            raw_norm_y = global_y / frame_h
                            
                            # Subtract the anchor
                            # This makes the wrist ALWAYS (0,0,0). 
                            # If a fingertip is at X: 0.6 and the wrist is at X: 0.5, the saved value is 0.1 (thus giving distance from tip to wrist)
                            final_x = raw_norm_x - wrist_norm_x
                            final_y = raw_norm_y - wrist_norm_y
                            final_z = landmark.z - wrist_z 
                            
                            extracted_points.extend([final_x, final_y, final_z])
                            
                            # Visual feedback using global to let the user know exactly what is appeearing
                            cv2.circle(frame, (global_x, global_y), 5, (0, 255, 0), -1)
                            
                        keypoints = np.array(extracted_points)
                        last_good_keypoints = keypoints
    
            window.add_frame(keypoints)
    
            # UI Overlay
            cv2.rectangle(frame, (0, 0), (frame_w, 40), (0, 0, 0), -1)
            status_text = f"FIXING: {action} | Overwriting File: {sequence}.npy | ESC to Quit"
            cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1, cv2.LINE_AA)
    
            if frame_num == 0: 
                cv2.putText(frame, f'RE-RECORDING FILE {sequence}...', (80, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                cv2.imshow('Data Correction', frame)
                
                for _ in range(20): 
                    if cv2.waitKey(100) & 0xFF == 27: 
                        print("Exiting during countdown...")
                        cleanup()
                        sys.exit() 
            else: 
                cv2.imshow('Data Correction', frame)
                if cv2.waitKey(10) & 0xFF == 27:
                    print("Exiting during capture...")
                    cleanup()
                    sys.exit()
                
        # Save directly over the corrupted file
        npy_path = os.path.join(DATA_PATH, str(sequence))
        np.save(npy_path, window.get_series())
        print(f"Successfully overwrote {sequence}.npy")
    
    cleanup()
    print("\nAll targeted sequences have been patched!")
