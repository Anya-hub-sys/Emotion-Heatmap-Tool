import cv2
import mediapipe as mp
import time
import csv
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mss
import tkinter as tk

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "emotion_training_data.csv"
MODEL_FILE = "emotion_model.pkl"

EMOTIONS = [
    "Happy",
    "Surprised",
    "Confused",
    "Suspicion",
    "Engaged",
    "Thinking",
    "Bored"
]

SMOOTHING_FACTOR = 0.10


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(blendshapes):

    scores = {
        b.category_name: b.score
        for b in blendshapes
    }

    feature_names = [
        "mouthSmileLeft",
        "mouthSmileRight",
        "browOuterUpLeft",
        "browOuterUpRight",
        "browDownLeft",
        "browDownRight",
        "jawOpen",
        "eyeSquintLeft",
        "eyeSquintRight",
        "eyeWideLeft",
        "eyeWideRight",
        "mouthPressLeft",
        "mouthPressRight",
        "jawForward",
        "eyeLookDownLeft",
        "eyeLookDownRight",
        "eyeBlinkLeft",
        "eyeBlinkRight",
        "eyeLookUpLeft",
        "eyeLookUpRight",
        "eyeLookInLeft",
        "eyeLookInRight",
        "eyeLookOutLeft",
        "eyeLookOutRight",
        "mouthFrownLeft",
        "mouthFrownRight",
        "mouthPucker",
        "mouthFunnel",
        "mouthStretchLeft",
        "mouthStretchRight"
    ]

    return [
        scores.get(name, 0.0)
        for name in feature_names
    ]


# ============================================================
# CAMERA / SCREEN CAPTURE
# ============================================================

def choose_input():

    print()
    print("Choose input source:")
    print("1 - Webcam")
    print("2 - Screen region")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "2":

        sct = mss.mss()
        selection = {}

        def on_mouse_down(event):
            selection["x1"] = event.x_root
            selection["y1"] = event.y_root

        def on_mouse_drag(event):

            if "x1" not in selection:
                return

            x1 = selection["x1"]
            y1 = selection["y1"]

            canvas.coords(
                rect,
                x1 - root.winfo_x(),
                y1 - root.winfo_y(),
                event.x_root - root.winfo_x(),
                event.y_root - root.winfo_y()
            )

        def on_mouse_up(event):

            selection["x2"] = event.x_root
            selection["y2"] = event.y_root

            root.quit()

        root = tk.Tk()

        root.attributes("-fullscreen", True)
        root.attributes("-alpha", 0.3)
        root.attributes("-topmost", True)

        canvas = tk.Canvas(
            root,
            cursor="cross",
            bg="grey"
        )

        canvas.pack(
            fill=tk.BOTH,
            expand=True
        )

        rect = canvas.create_rectangle(
            0,
            0,
            0,
            0,
            outline="red",
            width=3
        )

        canvas.bind(
            "<ButtonPress-1>",
            on_mouse_down
        )

        canvas.bind(
            "<B1-Motion>",
            on_mouse_drag
        )

        canvas.bind(
            "<ButtonRelease-1>",
            on_mouse_up
        )

        print()
        print(
            "Drag a box around the person's video tile "
            "and release the mouse."
        )

        root.mainloop()
        root.destroy()

        x1 = min(
            selection["x1"],
            selection["x2"]
        )

        y1 = min(
            selection["y1"],
            selection["y2"]
        )

        x2 = max(
            selection["x1"],
            selection["x2"]
        )

        y2 = max(
            selection["y1"],
            selection["y2"]
        )

        monitor = {
            "left": x1,
            "top": y1,
            "width": x2 - x1,
            "height": y2 - y1
        }

        return "screen", sct, monitor

    else:

        cap = cv2.VideoCapture(0)

        return "webcam", cap, None


# ============================================================
# GET FRAME
# ============================================================

def get_frame(source_type, source, monitor):

    if source_type == "webcam":

        success, frame = source.read()

        return success, frame

    screenshot = source.grab(monitor)

    frame = np.array(screenshot)

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGRA2BGR
    )

    return True, frame


# ============================================================
# DATA COLLECTION
# ============================================================

def collect_data():

    print()
    print("======================================")
    print("DATA COLLECTION MODE")
    print("======================================")
    print()
    print("The program will collect facial features.")
    print()
    print("Press:")
    print("H = Happy")
    print("U = Surprised")
    print("C = Confused")
    print("S = Suspicion")
    print("E = Engaged")
    print("T = Thinking")
    print("B = Bored")
    print("Q = Quit")
    print()
    print("Only press a key when the person's")
    print("current state matches that label.")
    print()

    source_type, source, monitor = choose_input()

    file_exists = os.path.exists(DATA_FILE)

    file = open(
        DATA_FILE,
        "a",
        newline=""
    )

    writer = csv.writer(file)

    feature_count = len(
        extract_features([])
    )

    if not file_exists:

        header = [
            "timestamp",
            "label"
        ]

        header += [
            f"feature_{i}"
            for i in range(feature_count)
        ]

        writer.writerow(header)

    current_label = None

    cv2.namedWindow(
        "Data Collection",
        cv2.WINDOW_NORMAL
    )

    while True:

        success, frame = get_frame(
            source_type,
            source,
            monitor
        )

        if not success:
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            time.time() * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        display = frame.copy()

        if result.face_blendshapes:

            features = extract_features(
                result.face_blendshapes[0]
            )

            if current_label is not None:

                timestamp = datetime.now().isoformat()

                writer.writerow([
                    timestamp,
                    current_label
                ] + features)

                file.flush()

            cv2.putText(
                display,
                f"Label: {current_label or 'NONE'}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                "H U C S E T B = labels | Q = quit",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                display,
                "NO FACE",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Data Collection",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("h"):
            current_label = "Happy"

        elif key == ord("u"):
            current_label = "Surprised"

        elif key == ord("c"):
            current_label = "Confused"

        elif key == ord("s"):
            current_label = "Suspicion"

        elif key == ord("e"):
            current_label = "Engaged"

        elif key == ord("t"):
            current_label = "Thinking"

        elif key == ord("b"):
            current_label = "Bored"

        elif key == ord("q"):
            break

    file.close()

    cv2.destroyAllWindows()

    if source_type == "webcam":
        source.release()

    print()
    print("Data collection finished.")
    print(f"Saved to: {DATA_FILE}")


# ============================================================
# TRAIN MACHINE LEARNING MODEL
# ============================================================

def train_model():

    print()
    print("======================================")
    print("TRAINING MODEL")
    print("======================================")

    if not os.path.exists(DATA_FILE):

        print()
        print("Training data does not exist.")
        print("Run data collection first.")
        return

    df = pd.read_csv(DATA_FILE)

    if len(df) < 100:

        print()
        print(
            f"Only {len(df)} samples found."
        )

        print(
            "Collect more data before training."
        )

        return

    print()
    print("Samples:", len(df))
    print()
    print("Samples per emotion:")

    print(
        df["label"].value_counts()
    )

    X = df.drop(
        columns=["timestamp", "label"]
    )

    y = df["label"]

    if len(y.unique()) < 2:

        print()
        print(
            "You need at least two different "
            "emotion labels."
        )

        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    print()
    print("Training...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print(
        f"Test accuracy: {accuracy * 100:.2f}%"
    )

    print()
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print()
    print(
        f"Model saved as: {MODEL_FILE}"
    )


# ============================================================
# LIVE MACHINE LEARNING
# ============================================================

def run_live():

    if not os.path.exists(MODEL_FILE):

        print()
        print("No trained model found.")
        print()
        print(
            "First collect data and train the model."
        )

        return

    model = joblib.load(
        MODEL_FILE
    )

    source_type, source, monitor = choose_input()

    smoothed = {
        emotion: 0.0
        for emotion in EMOTIONS
    }

    cv2.namedWindow(
        "Machine Learning Emotion Detection",
        cv2.WINDOW_NORMAL
    )

    log_file = open(
        "emotion_ml_log.csv",
        "w",
        newline=""
    )

    writer = csv.writer(
        log_file
    )

    writer.writerow(
        ["timestamp"] + EMOTIONS
    )

    while True:

        success, frame = get_frame(
            source_type,
            source,
            monitor
        )

        if not success:
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            time.time() * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        display = frame.copy()

        if result.face_blendshapes:

            features = extract_features(
                result.face_blendshapes[0]
            )

            X = pd.DataFrame(
                [features]
            )

            probabilities = model.predict_proba(
                X
            )[0]

            model_classes = model.classes_

            raw = {
                emotion: 0.0
                for emotion in EMOTIONS
            }

            for emotion, probability in zip(
                model_classes,
                probabilities
            ):

                if emotion in raw:
                    raw[emotion] = (
                        probability * 100
                    )

            for emotion in EMOTIONS:

                smoothed[emotion] = (
                    SMOOTHING_FACTOR *
                    raw[emotion]
                    +
                    (1 - SMOOTHING_FACTOR) *
                    smoothed[emotion]
                )

            emotions = {
                emotion: round(
                    smoothed[emotion],
                    1
                )
                for emotion in EMOTIONS
            }

            y_position = 35

            for emotion in EMOTIONS:

                text = (
                    f"{emotion}: "
                    f"{emotions[emotion]}%"
                )

                cv2.putText(
                    display,
                    text,
                    (30, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                y_position += 30

            timestamp = datetime.now().isoformat()

            writer.writerow(
                [timestamp]
                +
                [
                    emotions[e]
                    for e in EMOTIONS
                ]
            )

            log_file.flush()

        else:

            cv2.putText(
                display,
                "No face detected",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Machine Learning Emotion Detection",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    log_file.close()

    cv2.destroyAllWindows()

    if source_type == "webcam":
        source.release()

    print()
    print(
        "Live detection finished."
    )


# ============================================================
# GRAPH
# ============================================================

def show_graph():

    file_name = "emotion_ml_log.csv"

    if not os.path.exists(file_name):

        print()
        print(
            "No emotion log found."
        )

        return

    df = pd.read_csv(
        file_name
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    plt.figure(
        figsize=(14, 7)
    )

    for emotion in EMOTIONS:

        plt.plot(
            df["timestamp"],
            df[emotion],
            label=emotion
        )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "Model probability (%)"
    )

    plt.title(
        "Machine Learning Emotion Timeline"
    )

    plt.legend()

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        "emotion_ml_graph.png"
    )

    plt.show()


# ============================================================
# MAIN MENU
# ============================================================

def main():

    while True:

        print()
        print("======================================")
        print("FACIAL STATE MACHINE LEARNING SYSTEM")
        print("======================================")
        print()
        print("1 - Collect training data")
        print("2 - Train machine-learning model")
        print("3 - Run live detection")
        print("4 - Show emotion graph")
        print("5 - Exit")
        print()

        choice = input(
            "Choose an option: "
        ).strip()

        if choice == "1":

            collect_data()

        elif choice == "2":

            train_model()

        elif choice == "3":

            run_live()

        elif choice == "4":

            show_graph()

        elif choice == "5":

            break

        else:

            print(
                "Invalid choice."
            )


if __name__ == "__main__":

    main()