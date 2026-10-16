import cv2
import mediapipe as mp
import time
import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mss
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont
from collections import deque

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)
print("Choose input source:")
print("1 - Webcam")
print("2 - Screen region (for capturing someone's face during a call)")
choice = input("Enter 1 or 2: ")

use_screen_capture = (choice == "2")

border_window = None  # will hold the boundary overlay window if used


def show_capture_boundary(monitor, parent):
    """Opens a persistent, click-through, topmost window that draws a colored
    border exactly around the capture region so it's clear what's being tracked."""
    border = tk.Toplevel(parent)
    x, y = monitor["left"], monitor["top"]
    w, h = monitor["width"], monitor["height"]
    border.geometry(f"{w}x{h}+{x}+{y}")
    border.overrideredirect(True)
    border.attributes("-topmost", True)
    border.attributes("-transparentcolor", "black")
    border.config(bg="black")

    canvas = tk.Canvas(border, bg="black", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)
    thickness = 4
    canvas.create_rectangle(
        thickness // 2, thickness // 2,
        w - thickness // 2, h - thickness // 2,
        outline="red", width=thickness
    )
    return border


if use_screen_capture:
    sct = mss.mss()
    selection = {}

    def on_mouse_down(event):
        selection['x1'] = event.x_root
        selection['y1'] = event.y_root

    def on_mouse_drag(event):
        canvas.coords(rect, selection['x1'], selection['y1'], event.x_root - root.winfo_x(), event.y_root - root.winfo_y())

    def on_mouse_up(event):
        selection['x2'] = event.x_root
        selection['y2'] = event.y_root
        root.quit()

    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.attributes('-alpha', 0.3)
    root.attributes('-topmost', True)
    root.configure(bg='black')

    canvas = tk.Canvas(root, cursor="cross", bg='grey')
    canvas.pack(fill=tk.BOTH, expand=True)

    rect = canvas.create_rectangle(0, 0, 0, 0, outline='red', width=3)

    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_drag)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)

    print("Drag a box over the area you want to capture (e.g. the person's video tile), then release.")
    root.mainloop()

    x1, y1 = min(selection['x1'], selection['x2']), min(selection['y1'], selection['y2'])
    x2, y2 = max(selection['x1'], selection['x2']), max(selection['y1'], selection['y2'])
    monitor = {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}

    # Hide the fullscreen selection overlay but keep the root alive so we can
    # spawn the persistent boundary window as a child of it
    root.withdraw()
    border_window = show_capture_boundary(monitor, root)
else:
    cap = cv2.VideoCapture(0)

cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
log_file = open('emotion_log.csv', mode='w', newline='')
csv_writer = csv.writer(log_file)
EMOTION_LABELS = ['Happy', 'Surprised', 'Confused', 'Suspicion', 'Engaged', 'Thinking', 'Bored']
csv_writer.writerow(['timestamp'] + EMOTION_LABELS)

blink_history = deque(maxlen=15)  # ~0.5s of frames at 30fps, tune if needed


def classify_emotion(blendshapes):
    scores = {b.category_name: b.score for b in blendshapes}

    smile = (scores.get('mouthSmileLeft', 0) + scores.get('mouthSmileRight', 0)) / 2
    brow_up = (scores.get('browOuterUpLeft', 0) + scores.get('browOuterUpRight', 0)) / 2
    brow_down = (scores.get('browDownLeft', 0) + scores.get('browDownRight', 0)) / 2
    mouth_open = scores.get('jawOpen', 0)
    eye_squint = (scores.get('eyeSquintLeft', 0) + scores.get('eyeSquintRight', 0)) / 2
    eye_wide = (scores.get('eyeWideLeft', 0) + scores.get('eyeWideRight', 0)) / 2
    mouth_press = (scores.get('mouthPressLeft', 0) + scores.get('mouthPressRight', 0)) / 2
    jaw_clench = scores.get('jawForward', 0)
    brow_asymmetry = abs(scores.get('browDownLeft', 0) - scores.get('browDownRight', 0))
    eye_look_down = (scores.get('eyeLookDownLeft', 0) + scores.get('eyeLookDownRight', 0)) / 2
    eye_look_side = (scores.get('eyeLookOutLeft', 0) + scores.get('eyeLookOutRight', 0)
                      + scores.get('eyeLookInLeft', 0) + scores.get('eyeLookInRight', 0)) / 4
    eye_blink = (scores.get('eyeBlinkLeft', 0) + scores.get('eyeBlinkRight', 0)) / 2

    blink_history.append(eye_blink)
    sustained_droop = sum(blink_history) / len(blink_history)

    happy = smile
    surprised = (brow_up + mouth_open + eye_wide) / 3
    confused = (brow_down + eye_squint) / 2
    suspicion = (mouth_press + jaw_clench + brow_asymmetry) / 3

    thinking = (brow_down * 0.4 + eye_look_side * 0.6)

    mouth_slack = mouth_open * 0.3
    bored = (sustained_droop * 0.6 + mouth_slack * 0.4 + eye_look_down * 0.2)

    engaged = max(0, 1 - sustained_droop - eye_look_down - eye_look_side) * 0.5

    total = happy + surprised + confused + suspicion + engaged + thinking + bored
    if total == 0:
        return {label: 0 for label in EMOTION_LABELS}

    return {
        "Happy": round((happy / total) * 100, 1),
        "Surprised": round((surprised / total) * 100, 1),
        "Confused": round((confused / total) * 100, 1),
        "Suspicion": round((suspicion / total) * 100, 1),
        "Engaged": round((engaged / total) * 100, 1),
        "Thinking": round((thinking / total) * 100, 1),
        "Bored": round((bored / total) * 100, 1),
    }


frame_index = 0
while True:
    if use_screen_capture:
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        success = True
    else:
        success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    font = ImageFont.truetype("Lato-Bold.ttf", 28)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    if result.face_blendshapes:
        emotions = classify_emotion(result.face_blendshapes[0])
        y = 20
        for label in EMOTION_LABELS:
            pct = emotions[label]
            text = f"{label}: {pct}%"
            draw.text((30, y), text, font=font, fill=(0, 255, 0))
            y += 26

        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        csv_writer.writerow([timestamp_str] + [emotions[label] for label in EMOTION_LABELS])
    else:
        draw.text((30, 20), "No face", font=font, fill=(0, 255, 0))

    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    h, w = frame.shape[:2]
    aspect_ratio = w / h

    _, _, win_w, win_h = cv2.getWindowImageRect("Emotion Detection")
    if win_w > 0 and win_h > 0:
        if win_w / win_h > aspect_ratio:
            new_h = win_h
            new_w = int(win_h * aspect_ratio)
        else:
            new_w = win_w
            new_h = int(win_w / aspect_ratio)
        resized = cv2.resize(frame, (new_w, new_h))

        canvas_img = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        x_offset = (win_w - new_w) // 2
        y_offset = (win_h - new_h) // 2
        canvas_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        display_frame = canvas_img
    else:
        display_frame = frame

    cv2.imshow("Emotion Detection", display_frame)

    # Keep the boundary overlay window alive/refreshed each frame
    if use_screen_capture and border_window is not None:
        border_window.update()

    frame_index += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if not use_screen_capture:
    cap.release()
else:
    if border_window is not None:
        border_window.destroy()
    root.destroy()
cv2.destroyAllWindows()
log_file.close()

# ---- Interactive graph with clickable legend as toggle buttons ----
df = pd.read_csv('emotion_log.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

fig, ax = plt.subplots(figsize=(13, 7))
plt.subplots_adjust(bottom=0.3)

colors = plt.cm.tab10(np.linspace(0, 1, len(EMOTION_LABELS)))
lines = {}
for label, color in zip(EMOTION_LABELS, colors):
    line, = ax.plot(df['timestamp'], df[label], label=label, color=color)
    lines[label] = line

ax.set_xlabel('Time')
ax.set_ylabel('Percentage')
ax.set_title('Emotion Timeline')
plt.setp(ax.get_xticklabels(), rotation=45)

legend = ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.25),
    ncol=len(EMOTION_LABELS),
    frameon=False
)

legend_map = {}
for legend_line, legend_text, label in zip(legend.get_lines(), legend.get_texts(), EMOTION_LABELS):
    legend_line.set_picker(10)
    legend_text.set_picker(10)
    legend_map[legend_line] = label
    legend_map[legend_text] = label

def on_pick(event):
    label = legend_map.get(event.artist)
    if label is None:
        return
    line = lines[label]
    visible = not line.get_visible()
    line.set_visible(visible)

    alpha = 1.0 if visible else 0.2
    for legend_line, legend_text, lbl in zip(legend.get_lines(), legend.get_texts(), EMOTION_LABELS):
        if lbl == label:
            legend_line.set_alpha(alpha)
            legend_text.set_alpha(alpha)

    fig.canvas.draw_idle()

fig.canvas.mpl_connect('pick_event', on_pick)

plt.savefig('emotion_graph.png', bbox_inches='tight')
plt.show()