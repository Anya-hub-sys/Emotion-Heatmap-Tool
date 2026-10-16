# Emotion Heatmap

Tracks a viewer's face with MediaPipe while they watch a video, logs emotions with timestamps, and produces an engagement graph. Useful for video reviews: it shows when viewers smiled, got surprised, or lost interest.

## Files
- `main.py` - main app (webcam or screen capture, live emotion detection, logging)
- `emotion.py` - emotion detection logic
- `face_landmarker.task` - MediaPipe face model (required)
- `emotion_training_data.csv` - reference data for emotion detection
- `Lato-Bold.ttf` - font used for on-screen text and graph
- `emotion_log.csv` - output: emotions with timestamps
- `emotion_graph.png` - output: emotion timeline graph

## Requirements
- Windows, Python 3.10 or newer
- A webcam (a phone camera app like Iriun also works)

## Install
1. Download or clone the project.
2. Open a terminal in the project folder and create a virtual environment:
```
   python -m venv venv
   venv\Scripts\activate
```
3. Install the libraries:
```
   pip install opencv-python mediapipe numpy pandas matplotlib
```
   (Add any other libraries your code imports.)
4. Make sure `face_landmarker.task` is in the same folder as `main.py`.

## Run
```
python main.py
```
Choose an input:
- `1` - Webcam
- `2` - Screen region (to capture someone's face during a call)

Press `q` (or Esc) in the video window to stop. The log and graph are then saved.

## Using it for video reviews
1. Start the tool and have the viewer watch your video.
2. Stop the tool when the video ends.
3. Open `emotion_graph.png` and `emotion_log.csv`.
4. Match the timestamps to your video to find strong moments and slow sections.

## Tips
- Use good, even lighting and keep the face clearly visible.
- Get the viewer's consent before recording.
- Treat results as a guide, not proof. Talk to the viewer too.
- If the webcam hangs, use `cv2.VideoCapture(0, cv2.CAP_DSHOW)` and close apps using the camera.
