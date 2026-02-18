import cv2
import numpy as np
import face_recognition
import os
import pickle
import time

# --- CONFIGURATION ---
DATASET_PATH = 'dataset'
MODEL_FILE = 'trained_model.pkl'
CAPTURE_COUNT = 30  # Increased for better accuracy
TOLERANCE = 0.55    # Slightly relaxed for better recognition

def verify_and_capture():
    print("="*50 + "\nFACE RECOGNITION TRAINER (HIGH ACCURACY)\n" + "="*50)
    
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    # 1. Get Person Name
    person_name = input("\nEnter name of the person to train: ").strip()
    person_path = os.path.join(DATASET_PATH, person_name)
    
    if not os.path.exists(person_path):
        os.makedirs(person_path)
        print(f"Created new folder for {person_name}")
    
    # 2. Check Existing
    existing = len([f for f in os.listdir(person_path) if f.endswith('.jpg')])
    print(f"Found {existing} existing images.")

    # 3. Capture Phase
    print("\n" + "-"*50)
    print("CAPTURE PHASE - PREPARING")
    print("-" * 50)
    print("We will capture 30 images. Please follow instructions.")
    print("Lighting should be bright and even on your face.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No camera.")
        return

    captured = 0
    instructions = [
        (0, "Look straight at camera"),
        (10, "Turn head slightly LEFT"),
        (15, "Turn head slightly RIGHT"),
        (20, "Tilt head slightly UP"),
        (25, "Tilt head slightly DOWN")
    ]
    
    print("\nPress SPACE to start capturing...")
    while True:
        ret, frame = cap.read()
        cv2.imshow('Training', frame)
        if cv2.waitKey(1) == 32: break

    while captured < CAPTURE_COUNT:
        ret, frame = cap.read()
        if not ret: break
        
        display = frame.copy()
        
        # Determine current instruction
        current_instruction = instructions[0][1]
        for threshold, text in instructions:
            if captured >= threshold:
                current_instruction = text

        # UI
        cv2.putText(display, f"Count: {captured}/{CAPTURE_COUNT}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, current_instruction, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.rectangle(display, (0, 0), (640, 80), (0,0,0), -1)
        cv2.addWeighted(display, 0.3, frame, 0.7, 0, frame) # Transparency
        
        cv2.imshow('Training', display)
        
        # Auto-capture every 500ms (approx) to get variety
        time.sleep(0.2) 
        
        # Save image
        filename = f"{person_name}_{int(time.time())}_{captured}.jpg"
        cv2.imwrite(os.path.join(person_path, filename), frame)
        captured += 1
        
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()
    
    # 4. Training Phase
    print("\n" + "="*50)
    print("TRAINING MODEL...")
    
    known_encodings = []
    known_names = []

    # Process ALL folders in dataset
    for person in os.listdir(DATASET_PATH):
        p_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(p_path): continue
        
        print(f"Processing: {person}")
        for fname in os.listdir(p_path):
            if fname.endswith(('.jpg', '.png')):
                img = cv2.imread(os.path.join(p_path, fname))
                if img is None: continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb)
                if encs:
                    known_encodings.append(encs[0])
                    known_names.append(person)

    data = {'encodings': known_encodings, 'names': known_names}
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(data, f)
        
    print(f"\nModel saved! Total Faces: {len(known_names)}")

if __name__ == "__main__":
    verify_and_capture()