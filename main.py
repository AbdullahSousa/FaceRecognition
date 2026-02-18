import cv2
import face_recognition
import pickle
import numpy as np
import os

# --- CONFIG ---
MODEL_FILE = 'trained_model.pkl'
TOLERANCE = 0.50

def main():
    print("Loading Model...")
    if not os.path.exists(MODEL_FILE):
        print("Model not found. Please train first.")
        return

    with open(MODEL_FILE, 'rb') as f:
        data = pickle.load(f)
    known_encodings = data['encodings']
    known_names = data['names']
    
    cap = cv2.VideoCapture(0)
    print("Starting Camera... Press 'ESC' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Resize for speed (1/4th size)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 2. Detect all faces and calculate encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        # 3. Loop through EACH face found in the frame
        for face_encoding in face_encodings:
            # Calculate distance to all known faces
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
            name = "Unknown"
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    # Calculate confidence percentage
                    confidence = round((1 - face_distances[best_match_index]) * 100, 1)
                    name = f"{name} ({confidence}%)"

            face_names.append(name)

        # 4. Display Results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up (x4)
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw box
            color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Multi-Person Recognition', frame)

        if cv2.waitKey(1) == 27: # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()