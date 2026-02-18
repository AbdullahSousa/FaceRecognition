import cv2
import numpy as np
import face_recognition
import pickle
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_FILE = 'trained_model.pkl'
TOLERANCE = 0.50 # Strict tolerance

def load_model():
    if not os.path.exists(MODEL_FILE): return None, None
    with open(MODEL_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['encodings'], data['names']

def test_pipeline():
    known_encodings, known_names = load_model()
    if not known_encodings: return

    # Global lists to store results from ALL people tested in this session
    global_y_true = []
    global_y_pred = []
    
    testing_active = True
    
    while testing_active:
        print("\n" + "="*50)
        print("NEW TEST SUBJECT")
        print("="*50)
        unique_people = sorted(set(known_names))
        print("Available people:", ", ".join(unique_people))
        
        ground_truth = input("\nWho is in front of the camera now? (Exact name): ").strip()
        if ground_truth not in unique_people:
            print("Name not found. Skipping...")
            continue

        print(f"\nSTARTING TEST FOR: {ground_truth}")
        print("Collect 20-30 samples. Press 'q' to stop collecting for this person.")
        
        cap = cv2.VideoCapture(0)
        samples_collected = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            faces = face_recognition.face_locations(rgb_small)
            encodings = face_recognition.face_encodings(rgb_small, faces)
            
            detected_name = "Unknown"
            
            if faces:
                # Find best match
                matches = face_recognition.compare_faces(known_encodings, encodings[0], tolerance=TOLERANCE)
                dists = face_recognition.face_distance(known_encodings, encodings[0])
                best_match_idx = np.argmin(dists)
                
                if matches[best_match_idx]:
                    detected_name = known_names[best_match_idx]
                
                # Visuals
                top, right, bottom, left = faces[0]
                top, right, bottom, left = top*4, right*4, bottom*4, left*4
                color = (0, 255, 0) if detected_name == ground_truth else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, f"{detected_name}", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Auto-collect data
                global_y_true.append(ground_truth)
                global_y_pred.append(detected_name)
                samples_collected += 1

            cv2.putText(frame, f"Samples: {samples_collected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            cv2.putText(frame, "Press 'q' to finish this person", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
            cv2.imshow('Testing', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Check if user wants to test another person
        choice = input("\nDo you want to test another person? (y/n): ").lower()
        if choice != 'y':
            testing_active = False

    # --- FINAL CALCULATIONS ---
    print("\n" + "="*50)
    print("GLOBAL SESSION RESULTS")
    print("="*50)
    
    if len(global_y_true) == 0:
        print("No data collected.")
        return

    # Metrics
    labels = sorted(list(set(known_names + ["Unknown"])))
    acc = accuracy_score(global_y_true, global_y_pred)
    f1 = f1_score(global_y_true, global_y_pred, average='weighted', zero_division=0)
    prec = precision_score(global_y_true, global_y_pred, average='weighted', zero_division=0)
    
    print(f"Total Samples: {len(global_y_true)}")
    print(f"Global Accuracy:  {acc*100:.2f}%")
    print(f"Global F1 Score:  {f1:.4f}")
    print(f"Global Precision: {prec:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(global_y_true, global_y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Global Confusion Matrix\nAcc: {acc:.2f} | F1: {f1:.2f}')
    plt.show()

if __name__ == "__main__":
    test_pipeline()