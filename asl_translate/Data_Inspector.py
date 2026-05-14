"""
Data_Inspector.py
TL;DR This program is designed purely as added feature to allow the devs to look at the data just to double check the Data_CorrectionV2 script, to make sure that good data isn't being thrown away. Able to take
entire folders of .npy files and check through them for unusable data (frozen frames and zeroes) along for the shape of the data if disproportionate. 
Prompts the user to run correction on them if there is issues in the dataset.
"""

import numpy as np
import os

# CONFIG (DON'T CHANGE THIS)
action = input("Enter the letter folder to scan (e.g., A): ").upper()
DATA_PATH = os.path.join('ASL_Dataset', action)

if not os.path.exists(DATA_PATH):
    print(f"Directory {DATA_PATH} does not exist. Please check the path.")
    exit()

print(f"\n--- Batch Data Report for {DATA_PATH} ---")

clean_files = []
corrupted_files = []

# Gather and sort files numerically
files = [f for f in os.listdir(DATA_PATH) if f.endswith(".npy")]
files.sort(key=lambda x: int(x.replace('.npy', '')))

for filename in files:
    filepath = os.path.join(DATA_PATH, filename)
    
    try:
        data = np.load(filepath)
        
        # Check 1: Shape Validation
        if data.shape != (16, 63):
            print(f"[{filename:^8}] ERROR - Wrong shape: {data.shape}")
            corrupted_files.append(filename)
            continue
            
        zero_frames = []
        frozen_frames = []
        
        # Check 2: Pure Zeros (No data being recorded)
        for frame_num in range(data.shape[0]):
            if np.all(data[frame_num] == 0):
                zero_frames.append(frame_num)
                
        # Check 3: Frozen Frames (No data being recorded mid-collection)
        for i in range(1, data.shape[0]):
            if np.array_equal(data[i], data[i-1]):
                frozen_frames.append(i)
        
        # Grading the file
        if not zero_frames and not frozen_frames:
            print(f"[{filename:^8}] PASS - 100% Clean")
            clean_files.append(filename)
        else:
            error_msg = []
            if zero_frames: error_msg.append(f"Zeros at {zero_frames}")
            if frozen_frames: error_msg.append(f"Frozen at {frozen_frames}")
            print(f"[{filename:^8}] FAIL - {' | '.join(error_msg)}")
            corrupted_files.append(filename)
            
    except Exception as e:
        print(f"[{filename:^8}] ERROR - Could not read file: {e}")

# Summary of Read File(s)
print("\n" + "="*45)
print("FINAL BATCH SUMMARY")
print("="*45)
print(f"Total Files Scanned: {len(files)}")
print(f"Clean Files:         {len(clean_files)}")
print(f"Corrupted Files:     {len(corrupted_files)}")

if not corrupted_files:
    print("\nStatus: PERFECT. This dataset is cleared for training!")
else:
    print("\nStatus: ACTION REQUIRED. Run Data_correctionV2.py on the failed files.")
