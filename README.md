# Face Recognition

Real-time, multi-person face recognition from a webcam, built with Python, OpenCV and the `face_recognition` library.

The project covers the full workflow: collecting training images, building the face database, recognising people live, and measuring accuracy with a confusion matrix.

## How it works

| Script | Purpose |
|---|---|
| `train_model.py` | Asks for a person's name, captures 30 webcam images while guiding them to look straight, left, right, up and down, then encodes every face in `dataset/` and saves the encodings to `trained_model.pkl` |
| `main.py` | Opens the webcam and recognises every face in the frame. Known faces get a green box with the name and a confidence percentage; unknown faces get a red box |
| `test_model.py` | Evaluation mode: you say who is in front of the camera, it collects predictions, then reports accuracy, F1 score and precision and plots a confusion matrix |

**Details**

- Each face is turned into a 128-dimensional encoding and matched to the closest known encoding by face distance.
- A tolerance threshold (0.50 when recognising) decides whether the closest match is accepted or labelled **Unknown**.
- Frames are scaled down to a quarter size before detection to keep the video smooth.

## Results

The repository includes confusion matrices from testing on two people (`confusion_matrix_Abdullah.png`, `confusion_matrix_Naser.png`).

## Getting started

```bash
pip install opencv-python face_recognition numpy scikit-learn matplotlib seaborn

python train_model.py   # add a person (repeat for each person)
python main.py          # live recognition, press ESC to exit
python test_model.py    # measure accuracy
```

`face_recognition` depends on `dlib`, which needs CMake and a C++ build toolchain to install.

## Tech stack

Python · OpenCV · face_recognition (dlib) · NumPy · scikit-learn · Matplotlib · Seaborn
