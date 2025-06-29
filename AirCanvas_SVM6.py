import cv2
import numpy as np
import mediapipe as mp
import math
import random
import time
import pickle
from collections import deque
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pyttsx3
import threading
import queue

# ==================== GLOBALS & INITIAL SETUP =====================

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Queue for non-blocking voice feedback
speech_queue = queue.Queue()

def speak_thread():
    while True:
        try:
            message = speech_queue.get(block=True)
            if message == "STOP":
                break
            engine.say(message)
            engine.runAndWait()
            speech_queue.task_done()
        except queue.Empty:
            continue

# Start the speech thread
speech_thread = threading.Thread(target=speak_thread, daemon=True)
speech_thread.start()

# Unified strokes history for freehand, shape, and text drawings
strokes = []
redo_strokes = []
current_stroke = None

# Text input state
text_input_mode = False
text_input = ""
text_input_position = None

# Flag to enable/disable drawing via gestures
drawing_enabled = True

# Brush sizes and types
brushSizes = [5, 10, 20, 30]
brushSizeIndex = 0

brush_types = ["round", "square", "dotted", "spray", "pattern", "star"]
brush_type_index = 0
brush_type = brush_types[brush_type_index]

# Preset colours (BGR)
preset_colors = {
    "blue":   (255, 0, 0),
    "green":  (0, 255, 0),
    "red":    (0, 0, 255),
    "yellow": (0, 255, 255),
    "eraser": (255, 255, 255)
}

selected_mode = "blue"
selected_color = preset_colors["blue"]

# Drawing mode selector
drawing_mode = "freehand"
shape_drawing_active = False
shape_start = None
shape_type = "rectangle"

# Gesture Timing & Swipe Variables
last_gesture_time = 0
gesture_delay = 1.0
two_finger_start = None
swipe_threshold = 100

# Sign-to-Text Variables
sign_to_text_mode = False
recognized_text = ""
last_sign_time = 0
last_sign_letter = ""
sign_delay = 1.0

# Training Mode Flag
training_mode = False

# EMA Filter for smoothing fingertip
class EMAFilter:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.last_x, self.last_y = None, None

    def update(self, x, y):
        if self.last_x is None or self.last_y is None:
            self.last_x, self.last_y = x, y
        else:
            self.last_x = self.alpha * x + (1 - self.alpha) * self.last_x
            self.last_y = self.alpha * y + (1 - self.alpha) * self.last_y
        return int(self.last_x), int(self.last_y)

ema_filter = EMAFilter()

# Gesture Feedback Variables
feedback_message = ""
feedback_start_time = 0
feedback_duration = 0.5

# Brush Preview Variables
preview_size = 100
brush_preview = np.ones((preview_size, preview_size, 3), dtype=np.uint8) * 255
brush_preview_changed = True

# Canvas Redraw Optimization
canvas_needs_redraw = True

# Help Menu
help_text = [
    "Air Canvas Help Menu:",
    "1. Use your hand to draw in the air.",
    "2. In SHAPE mode, spread index & middle fingers to start shape,",
    "   then bring them together to complete it.",
    "3. Press '1' for Rectangle, '2' for Circle, '3' for Triangle.",
    "4. Press 'CLEAR' to clear the canvas (or do thumbs down gesture).",
    "5. Press 'BLUE', 'GREEN', 'RED', 'YELLOW' to use preset colours.",
    "6. Press 'COLOR' to pick a custom colour.",
    "7. Press 'ERASER' to erase.",
    "8. Press 'SIZE' to change brush size.",
    "9. Press 'MODE' to toggle drawing modes.",
    "10. Press 'b' to change brush type.",
    "11. Press 's' or show peace sign to save the canvas.",
    "12. Press 'd' to enter text input mode (type, press Enter to draw).",
    "13. In Sign-to-Text mode, use clenched fist to draw recognized text.",
    "14. Press 'q' to quit.",
    "15. Gesture commands:",
    "    Thumbs UP: start drawing mode",
    "    Open Palm: stop drawing mode",
    "    Two-finger swipe left: undo last stroke",
    "    Two-finger swipe right: redo last stroke",
    "    Clenched Fist (in Sign-to-Text): draw text",
    "16. **Sign-to-Text Mode** (toggle with 't'):",
    "    Recognizes ASL letters using SVM.",
    "17. Press 'p' to toggle Training Mode for ASL data collection.",
    "18. In Training Mode, press 'a'-'z' to collect ASL letter data."
]
show_help = False

# Canvas & Frame Setup
canvas_width = 640
canvas_height = 480
paintWindow = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

# Toolbar Setup
toolbar_height = 41
num_buttons = 9
button_width = canvas_width // num_buttons

toolbar_buttons = [
    {"label": "CLEAR"},
    {"label": "BLUE",   "mode": "blue"},
    {"label": "GREEN",  "mode": "green"},
    {"label": "RED",    "mode": "red"},
    {"label": "YELLOW", "mode": "yellow"},
    {"label": "COLOR"},
    {"label": "ERASER", "mode": "eraser"},
    {"label": "SIZE"},
    {"label": "MODE"}
]

# ------------------- Toolbar Drawing Function -------------------
def draw_toolbar(img):
    for i, button in enumerate(toolbar_buttons):
        x1 = i * button_width
        y1 = 0
        x2 = (i + 1) * button_width
        y2 = toolbar_height
        if button["label"] == "BLUE":
            border_color = preset_colors["blue"]
        elif button["label"] == "GREEN":
            border_color = preset_colors["green"]
        elif button["label"] == "RED":
            border_color = preset_colors["red"]
        elif button["label"] == "YELLOW":
            border_color = preset_colors["yellow"]
        elif button["label"] == "COLOR":
            border_color = (128, 0, 128)
        elif button["label"] == "ERASER":
            border_color = (128, 128, 128)
        elif button["label"] == "MODE":
            border_color = (0, 165, 255)
        else:
            border_color = (0, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 2)
        text_size, _ = cv2.getTextSize(button["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = x1 + (button_width - text_size[0]) // 2
        text_y = y1 + (toolbar_height + text_size[1]) // 2
        text_color = (0, 0, 0) if button["label"] != "COLOR" else (255, 255, 255)
        cv2.putText(img, button["label"], (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

# ------------------- Color Picker Function -------------------
def open_color_picker():
    global brush_preview_changed
    cv2.namedWindow("Color Picker")
    def nothing(x):
        pass
    cv2.createTrackbar("R", "Color Picker", 0, 255, nothing)
    cv2.createTrackbar("G", "Color Picker", 0, 255, nothing)
    cv2.createTrackbar("B", "Color Picker", 0, 255, nothing)
    cv2.setTrackbarPos("R", "Color Picker", selected_color[2])
    cv2.setTrackbarPos("G", "Color Picker", selected_color[1])
    cv2.setTrackbarPos("B", "Color Picker", selected_color[0])
    
    while True:
        picker_img = np.zeros((100, 300, 3), np.uint8)
        r = cv2.getTrackbarPos("R", "Color Picker")
        g = cv2.getTrackbarPos("G", "Color Picker")
        b = cv2.getTrackbarPos("B", "Color Picker")
        picker_img[:] = (b, g, r)
        cv2.imshow("Color Picker", picker_img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            break
    cv2.destroyWindow("Color Picker")
    brush_preview_changed = True
    return (b, g, r)

# ------------------- Brush Preview Function -------------------
def update_brush_preview():
    global brush_preview, brush_preview_changed
    if not brush_preview_changed:
        return
    brush_preview[:] = 255
    preview_points = [(preview_size // 4, preview_size // 2), (3 * preview_size // 4, preview_size // 2)]
    draw_stroke(brush_preview, preview_points, selected_color, brushSizes[brushSizeIndex], brush_type)
    brush_preview_changed = False

# ------------------- Brush Drawing Function -------------------
def draw_stroke(frame, stroke_points, color, brush_size, brush_style):
    if brush_style == "round":
        for i in range(1, len(stroke_points)):
            if stroke_points[i - 1] is None or stroke_points[i] is None:
                continue
            cv2.line(frame, stroke_points[i - 1], stroke_points[i], color, brush_size)
    elif brush_style == "square":
        half = brush_size // 2
        for pt in stroke_points:
            if pt is None:
                continue
            top_left = (pt[0] - half, pt[1] - half)
            bottom_right = (pt[0] + half, pt[1] + half)
            cv2.rectangle(frame, top_left, bottom_right, color, -1)
    elif brush_style == "dotted":
        for pt in stroke_points:
            if pt is None:
                continue
            cv2.circle(frame, pt, brush_size // 2, color, -1)
    elif brush_style == "spray":
        for pt in stroke_points:
            if pt is None:
                continue
            for _ in range(10):
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0, brush_size)
                offset_x = int(radius * math.cos(angle))
                offset_y = int(radius * math.sin(angle))
                spray_pt = (pt[0] + offset_x, pt[1] + offset_y)
                cv2.circle(frame, spray_pt, 1, color, -1)
    elif brush_style == "pattern":
        offset = brush_size // 2
        for pt in stroke_points:
            if pt is None:
                continue
            pt1 = (pt[0] - offset, pt[1] - offset)
            pt2 = (pt[0] + offset, pt[1] + offset)
            pt3 = (pt[0] - offset, pt[1] + offset)
            pt4 = (pt[0] + offset, pt[1] - offset)
            cv2.line(frame, pt1, pt2, color, 1)
            cv2.line(frame, pt3, pt4, color, 1)
    elif brush_style == "star":
        for pt in stroke_points:
            if pt is None:
                continue
            star_radius = brush_size
            points = []
            for i in range(5):
                angle = math.radians(90 + i * 72)
                x = int(pt[0] + star_radius * math.cos(angle))
                y = int(pt[1] - star_radius * math.sin(angle))
                points.append((x, y))
            for i in range(5):
                pt_a = points[i]
                pt_b = points[(i + 2) % 5]
                cv2.line(frame, pt_a, pt_b, color, 1)

# ------------------- Redraw Canvas Function -------------------
def redraw_canvas():
    global paintWindow, canvas_needs_redraw
    if not canvas_needs_redraw:
        return
    paintWindow[toolbar_height:,:,:] = 255
    for stroke in strokes:
        if stroke["type"] == "freehand":
            draw_stroke(paintWindow, stroke["points"], stroke["color"],
                        stroke["brush_size"], stroke["brush_type"])
        elif stroke["type"] == "shape":
            if stroke["shape_type"] == "rectangle":
                cv2.rectangle(paintWindow, stroke["start"], stroke["end"],
                              stroke["color"], stroke["brush_size"])
            elif stroke["shape_type"] == "circle":
                center_circle = ((stroke["start"][0] + stroke["end"][0]) // 2,
                                 (stroke["start"][1] + stroke["end"][1]) // 2)
                radius = int(np.hypot(stroke["end"][0] - stroke["start"][0],
                                      stroke["end"][1] - stroke["start"][1]) / 2)
                cv2.circle(paintWindow, center_circle, radius,
                           stroke["color"], stroke["brush_size"])
            elif stroke["shape_type"] == "triangle":
                pt1 = stroke["start"]
                pt2 = stroke["end"]
                mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                base_length = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
                h = int(math.sqrt(3) * base_length / 2)
                pt3 = (mid[0], mid[1] - h)
                pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
                cv2.polylines(paintWindow, [pts], isClosed=True,
                              color=stroke["color"], thickness=stroke["brush_size"])
        elif stroke["type"] == "text":
            text_size = stroke["brush_size"] / 20.0
            cv2.putText(paintWindow, stroke["text"], stroke["position"],
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, stroke["color"],
                        max(1, int(stroke["brush_size"] / 10)), cv2.LINE_AA)
    if current_stroke is not None and current_stroke["type"] == "freehand":
        draw_stroke(paintWindow, current_stroke["points"], current_stroke["color"],
                    current_stroke["brush_size"], current_stroke["brush_type"])
    canvas_needs_redraw = False

# ------------------- SVM-Based ASL Recognition -------------------
def collect_data(letter, landmarks, filename="asl_data.pkl"):
    data_point = {"letter": letter, "landmarks": landmarks}
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        data = []
    data.append(data_point)
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved data for letter: {letter}")

def train_svm_model(filename="asl_data.pkl", model_filename="asl_svm_model.pkl"):
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("No training data found. Collect data first.")
        return None

    X = []
    y = []
    for entry in data:
        wrist = np.array(entry["landmarks"][0])
        features = []
        for lm in entry["landmarks"]:
            norm_lm = np.array(lm) - wrist
            features.extend([norm_lm[0], norm_lm[1]])
        X.append(features)
        y.append(entry["letter"])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = svm.SVC(kernel="rbf", probability=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Model Accuracy: {accuracy * 100:.2f}%")

    with open(model_filename, "wb") as f:
        pickle.dump(clf, f)
    return clf

try:
    with open("asl_svm_model.pkl", "rb") as f:
        clf = pickle.load(f)
except FileNotFoundError:
    clf = None
    print("SVM model not found. Train the model by collecting data and pressing 'm'.")

def recognize_asl_sign_ml(landmarks):
    if clf is None:
        return "", 0.0
    wrist = np.array(landmarks[0])
    features = []
    for lm in landmarks:
        norm_lm = np.array(lm) - wrist
        features.extend([norm_lm[0], norm_lm[1]])
    features = np.array([features])
    prediction = clf.predict(features)[0]
    confidence = clf.predict_proba(features)[0].max()
    return prediction, confidence

# ------------------- Initialize MediaPipe Hands -------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# ------------------- Initialize Webcam -------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, canvas_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, canvas_height)
ret = True

# Variable to ensure one toolbar button press per entry
toolbar_button_pressed = False

# ------------------- MAIN LOOP -------------------
while ret:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    frame = cv2.flip(frame, 1)
    small_frame = cv2.resize(frame, (320, 240))
    framergb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    frame = cv2.resize(frame, (canvas_width, canvas_height))
    draw_toolbar(frame)
    draw_toolbar(paintWindow)

    result = hands.process(framergb)
    landmarks = []
    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * canvas_width)
                lmy = int(lm.y * canvas_height)
                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        
        x_coords = [pt[0] for pt in landmarks]
        y_coords = [pt[1] for pt in landmarks]
        x_min = min(x_coords) - 10
        y_min = min(y_coords) - 10
        x_max = max(x_coords) + 10
        y_max = max(y_coords) + 10
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        fore_finger = ema_filter.update(landmarks[8][0], landmarks[8][1])
        center = fore_finger

        # Gesture-based Commands
        if current_time - last_gesture_time > gesture_delay:
            if not sign_to_text_mode:  # Suppress drawing gestures in sign-to-text mode
                if (landmarks[4][1] < landmarks[3][1] - 20) and (landmarks[4][1] < landmarks[0][1]) and \
                   (landmarks[8][1] > landmarks[6][1] and landmarks[12][1] > landmarks[10][1] and
                    landmarks[16][1] > landmarks[14][1] and landmarks[20][1] > landmarks[18][1]):
                    drawing_enabled = True
                    feedback_message = "Drawing Enabled!"
                    feedback_start_time = current_time
                    speech_queue.put("Drawing Enabled")
                    print("Drawing mode started via gesture (Thumbs Up)")
                    last_gesture_time = current_time
                elif (landmarks[8][1] < landmarks[6][1] and
                      landmarks[12][1] < landmarks[10][1] and
                      landmarks[16][1] < landmarks[14][1] and
                      landmarks[20][1] < landmarks[18][1]):
                    drawing_enabled = False
                    feedback_message = "Drawing Stopped!"
                    feedback_start_time = current_time
                    speech_queue.put("Drawing Stopped")
                    print("Drawing mode stopped via gesture (Open Palm)")
                    last_gesture_time = current_time
                elif (landmarks[4][1] > landmarks[3][1] + 20) and (landmarks[4][1] > landmarks[0][1]) and \
                     (landmarks[8][1] > landmarks[6][1] and landmarks[12][1] > landmarks[10][1] and
                      landmarks[16][1] > landmarks[14][1] and landmarks[20][1] > landmarks[18][1]):
                    strokes = []
                    redo_strokes = []
                    current_stroke = None
                    paintWindow[toolbar_height:,:,:] = 255
                    feedback_message = "Canvas Cleared!"
                    feedback_start_time = current_time
                    speech_queue.put("Canvas Cleared")
                    canvas_needs_redraw = True
                    print("Canvas cleared via gesture (Thumbs Down)")
                    last_gesture_time = current_time

            def is_two_finger(landmarks):
                return (landmarks[8][1] < landmarks[6][1] and
                        landmarks[12][1] < landmarks[10][1] and
                        landmarks[16][1] > landmarks[14][1] and
                        landmarks[20][1] > landmarks[18][1])
            def is_peace(landmarks):
                if is_two_finger(landmarks):
                    dist = np.linalg.norm(np.array(landmarks[8]) - np.array(landmarks[12]))
                    if dist > 40:
                        return True
                return False
            def is_clenched_fist(landmarks):
                # All fingertips below their respective PIP joints
                return (landmarks[8][1] > landmarks[6][1] + 20 and  # Index
                        landmarks[12][1] > landmarks[10][1] + 20 and  # Middle
                        landmarks[16][1] > landmarks[14][1] + 20 and  # Ring
                        landmarks[20][1] > landmarks[18][1] + 20 and  # Pinky
                        landmarks[4][1] > landmarks[3][1] + 20)  # Thumb

            if not sign_to_text_mode:  # Suppress save gesture in sign-to-text mode
                if is_peace(landmarks):
                    cv2.imwrite("canvas_output.png", paintWindow)
                    feedback_message = "Canvas Saved!"
                    feedback_start_time = current_time
                    speech_queue.put("Canvas Saved")
                    print("Drawing saved via gesture (Peace Sign)")
                    last_gesture_time = current_time
            else:  # Handle text drawing in sign-to-text mode
                if is_clenched_fist(landmarks) and recognized_text:
                    stroke = {
                        "type": "text",
                        "text": recognized_text,
                        "position": center,
                        "color": selected_color,
                        "brush_size": brushSizes[brushSizeIndex]
                    }
                    strokes.append(stroke)
                    redo_strokes = []
                    canvas_needs_redraw = True
                    feedback_message = "Text Drawn!"
                    feedback_start_time = current_time
                    speech_queue.put("Text Drawn")
                    print(f"Text drawn: {recognized_text}")
                    recognized_text = ""
                    last_gesture_time = current_time
                elif is_clenched_fist(landmarks) and not recognized_text:
                    feedback_message = "No Text to Draw!"
                    feedback_start_time = current_time
                    speech_queue.put("No Text to Draw")
                    last_gesture_time = current_time

            # Undo/Redo gestures
            if is_two_finger(landmarks) and not is_peace(landmarks):
                two_center = ((landmarks[8][0] + landmarks[12][0]) // 2,
                              (landmarks[8][1] + landmarks[12][1]) // 2)
                if two_finger_start is None:
                    two_finger_start = two_center
                else:
                    dx = two_center[0] - two_finger_start[0]
                    if dx < -swipe_threshold:
                        if strokes:
                            redo_strokes.append(strokes.pop())
                            feedback_message = "Stroke Undone!"
                            feedback_start_time = current_time
                            speech_queue.put("Stroke Undone")
                            canvas_needs_redraw = True
                            print("Undo gesture detected: Last stroke undone")
                            last_gesture_time = current_time
                        two_finger_start = None
                    elif dx > swipe_threshold:
                        if redo_strokes:
                            strokes.append(redo_strokes.pop())
                            feedback_message = "Stroke Redone!"
                            feedback_start_time = current_time
                            speech_queue.put("Stroke Redone")
                            canvas_needs_redraw = True
                            print("Redo gesture detected: Last stroke redone")
                            last_gesture_time = current_time
                        two_finger_start = None
            else:
                two_finger_start = None

        # Sign-to-Text Recognition with SVM
        if sign_to_text_mode and not text_input_mode:
            letter, confidence = recognize_asl_sign_ml(landmarks)
            if letter and letter != last_sign_letter and confidence > 0.7 and (current_time - last_sign_time > sign_delay):
                recognized_text += letter
                last_sign_time = current_time
                last_sign_letter = letter
                speech_queue.put(f"Letter {letter}")
                print(f"Recognized letter: {letter} (Confidence: {confidence:.2f})")

        # Toolbar Interaction
        if center[1] <= toolbar_height:
            if current_stroke is not None:
                strokes.append(current_stroke)
                current_stroke = None
                canvas_needs_redraw = True
            if not toolbar_button_pressed:
                button_index = center[0] // button_width
                if button_index < len(toolbar_buttons):
                    btn = toolbar_buttons[button_index]
                    if btn["label"] == "CLEAR":
                        strokes = []
                        redo_strokes = []
                        current_stroke = None
                        paintWindow[toolbar_height:,:,:] = 255
                        canvas_needs_redraw = True
                        brush_preview_changed = True
                        feedback_message = "Canvas Cleared!"
                        feedback_start_time = current_time
                        speech_queue.put("Canvas Cleared")
                        print("Canvas cleared via toolbar")
                    elif btn["label"] in ["BLUE", "GREEN", "RED", "YELLOW"]:
                        selected_mode = btn["mode"]
                        selected_color = preset_colors[selected_mode]
                        brush_preview_changed = True
                        feedback_message = f"{btn['label']} Selected!"
                        feedback_start_time = current_time
                        speech_queue.put(f"{btn['label']} Selected")
                        print(f"Color set to: {selected_mode.upper()}")
                    elif btn["label"] == "COLOR":
                        custom = open_color_picker()
                        selected_mode = "custom"
                        selected_color = custom
                        brush_preview_changed = True
                        feedback_message = "Custom Color Selected!"
                        feedback_start_time = current_time
                        speech_queue.put("Custom Color Selected")
                        print("Custom color selected")
                    elif btn["label"] == "ERASER":
                        selected_mode = "eraser"
                        selected_color = preset_colors["eraser"]
                        brush_preview_changed = True
                        feedback_message = "Eraser Selected!"
                        feedback_start_time = current_time
                        speech_queue.put("Eraser Selected")
                        print("Eraser selected")
                    elif btn["label"] == "SIZE":
                        brushSizeIndex = (brushSizeIndex + 1) % len(brushSizes)
                        brush_preview_changed = True
                        feedback_message = f"Brush Size {brushSizes[brushSizeIndex]}!"
                        feedback_start_time = current_time
                        speech_queue.put(f"Brush Size {brushSizes[brushSizeIndex]}")
                        print(f"Brush size changed to: {brushSizes[brushSizeIndex]}")
                    elif btn["label"] == "MODE":
                        if drawing_mode == "freehand":
                            drawing_mode = "shape"
                            feedback_message = "Shape Mode!"
                            feedback_start_time = current_time
                            speech_queue.put("Shape Mode")
                            print("Drawing mode changed to: SHAPE mode")
                        else:
                            drawing_mode = "freehand"
                            shape_drawing_active = False
                            shape_start = None
                            feedback_message = "Freehand Mode!"
                            feedback_start_time = current_time
                            speech_queue.put("Freehand Mode")
                            print("Drawing mode changed to: FREEHAND mode")
                toolbar_button_pressed = True
        else:
            toolbar_button_pressed = False

        # Drawing Based on Selected Mode
        if center[1] > toolbar_height and not text_input_mode:
            if drawing_mode == "freehand" and not sign_to_text_mode:
                if drawing_enabled:
                    if current_stroke is None:
                        current_stroke = {
                            "type": "freehand",
                            "points": [center],
                            "color": selected_color,
                            "brush_size": brushSizes[brushSizeIndex],
                            "brush_type": brush_type
                        }
                    else:
                        current_stroke["points"].append(center)
                    canvas_needs_redraw = True
            elif drawing_mode == "shape":
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                distance_val = np.hypot(index_tip[0] - middle_tip[0],
                                        index_tip[1] - middle_tip[1])
                if distance_val > 50:
                    shape_drawing_active = True
                    if shape_start is None:
                        shape_start = center
                else:
                    if shape_drawing_active and shape_start is not None:
                        shape_end = center
                        stroke = {
                            "type": "shape",
                            "shape_type": shape_type,
                            "start": shape_start,
                            "end": shape_end,
                            "color": selected_color,
                            "brush_size": brushSizes[brushSizeIndex]
                        }
                        strokes.append(stroke)
                        shape_drawing_active = False
                        shape_start = None
                        canvas_needs_redraw = True
                    if shape_drawing_active and shape_start is not None:
                        shape_end = center
                        if shape_type == "rectangle":
                            cv2.rectangle(frame, shape_start, shape_end, selected_color,
                                          brushSizes[brushSizeIndex])
                        elif shape_type == "circle":
                            center_circle = ((shape_start[0] + shape_end[0]) // 2,
                                             (shape_start[1] + shape_end[1]) // 2)
                            radius = int(np.hypot(shape_end[0] - shape_start[0],
                                                  shape_end[1] - shape_start[1]) / 2)
                            cv2.circle(frame, center_circle, radius, selected_color,
                                       brushSizes[brushSizeIndex])
                        elif shape_type == "triangle":
                            pt1 = shape_start
                            pt2 = shape_end
                            mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                            base_length = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
                            h = int(math.sqrt(3) * base_length / 2)
                            pt3 = (mid[0], mid[1] - h)
                            pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
                            cv2.polylines(frame, [pts], isClosed=True, color=selected_color,
                                          thickness=brushSizes[brushSizeIndex])
        cv2.circle(frame, center, 3, (0, 255, 0), -1)
    else:
        if drawing_mode == "freehand" and current_stroke is not None and not sign_to_text_mode:
            if len(current_stroke["points"]) > 1:
                strokes.append(current_stroke)
                redo_strokes = []
                canvas_needs_redraw = True
            current_stroke = None

    # Update and Draw Brush Preview
    update_brush_preview()
    frame[toolbar_height:toolbar_height + preview_size, canvas_width - preview_size:canvas_width, :] = brush_preview

    # Redraw Canvas if Needed
    redraw_canvas()

    # Display Text Input
    if text_input_mode:
        cv2.putText(frame, f"Typing: {text_input}", (10, canvas_height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Display Gesture Feedback
    if feedback_message and (current_time - feedback_start_time < feedback_duration):
        cv2.putText(frame, feedback_message, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        feedback_message = ""

    # Display Training Mode Status
    if training_mode:
        cv2.putText(frame, "Training Mode ON", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    if sign_to_text_mode:
        cv2.putText(paintWindow, "Text: " + recognized_text, (10, canvas_height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Sign-to-Text Mode ON", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Brush Size: {brushSizes[brushSizeIndex]}", (10, canvas_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Brush: {brush_type.upper()}", (10, canvas_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Mode: {drawing_mode.upper()}", (canvas_width - 120, canvas_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    if show_help:
        y_offset = toolbar_height + 10
        for i, line in enumerate(help_text):
            cv2.putText(frame, line, (10, y_offset + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1) & 0xFF
    if text_input_mode:
        if key in range(32, 127):  # Printable ASCII (letters, numbers, symbols)
            text_input += chr(key)
        elif key == 8:  # Backspace
            text_input = text_input[:-1]
        elif key == 13:  # Enter
            if text_input and text_input_position:
                stroke = {
                    "type": "text",
                    "text": text_input,
                    "position": text_input_position,
                    "color": selected_color,
                    "brush_size": brushSizes[brushSizeIndex]
                }
                strokes.append(stroke)
                redo_strokes = []
                canvas_needs_redraw = True
                feedback_message = "Text Drawn!"
                feedback_start_time = current_time
                speech_queue.put("Text Drawn")
                print(f"Text drawn: {text_input}")
            text_input_mode = False
            text_input = ""
            text_input_position = None
        elif key == 27:  # Escape
            text_input_mode = False
            text_input = ""
            text_input_position = None
            feedback_message = "Text Input Cancelled!"
            feedback_start_time = current_time
            speech_queue.put("Text Input Cancelled")
    else:
        # Toggle Training Mode
        if key == ord('p'):
            training_mode = not training_mode
            feedback_message = "Training Mode ON" if training_mode else "Training Mode OFF"
            feedback_start_time = current_time
            speech_queue.put(feedback_message)
            print(feedback_message)
        # Collect training data when in training mode
        elif training_mode and key in range(ord('a'), ord('z') + 1) and landmarks:
            letter = chr(key).upper()
            collect_data(letter, landmarks)
        # Existing functionalities when not in training mode
        elif not training_mode:
            if key == ord('s'):
                cv2.imwrite("canvas_output.png", paintWindow)
                feedback_message = "Canvas Saved!"
                feedback_start_time = current_time
                speech_queue.put("Canvas Saved")
                print("Canvas saved as canvas_output.png")
            elif key == ord('h'):
                show_help = not show_help
            elif key == ord('t'):
                sign_to_text_mode = not sign_to_text_mode
                feedback_message = "Sign-to-Text Mode ON" if sign_to_text_mode else "Sign-to-Text Mode OFF"
                feedback_start_time = current_time
                speech_queue.put(feedback_message)
                print(feedback_message)
            elif key == ord('d') and landmarks:
                text_input_mode = True
                text_input_position = center
                feedback_message = "Enter Text Mode!"
                feedback_start_time = current_time
                speech_queue.put("Enter Text Mode")
                print("Text input mode activated")
            elif key == ord('1'):
                shape_type = "rectangle"
                feedback_message = "Rectangle Shape Selected!"
                feedback_start_time = current_time
                speech_queue.put("Rectangle Shape Selected")
                print("Shape set to rectangle")
            elif key == ord('2'):
                shape_type = "circle"
                feedback_message = "Circle Shape Selected!"
                feedback_start_time = current_time
                speech_queue.put("Circle Shape Selected")
                print("Shape set to circle")
            elif key == ord('3'):
                shape_type = "triangle"
                feedback_message = "Triangle Shape Selected!"
                feedback_start_time = current_time
                speech_queue.put("Triangle Shape Selected")
                print("Shape set to triangle")
            elif key == ord('b'):
                if current_stroke is not None:
                    strokes.append(current_stroke)
                    current_stroke = None
                    canvas_needs_redraw = True
                brush_type_index = (brush_type_index + 1) % len(brush_types)
                brush_type = brush_types[brush_type_index]
                brush_preview_changed = True
                feedback_message = f"Brush Type {brush_type.capitalize()}!"
                feedback_start_time = current_time
                speech_queue.put(f"Brush Type {brush_type.capitalize()}")
                print(f"Brush type changed to: {brush_type}")
            elif key == ord('q'):
                feedback_message = "Exiting Application!"
                feedback_start_time = current_time
                speech_queue.put("Exiting Application")
                break
            elif key in range(ord('a'), ord('z') + 1) and landmarks:
                letter = chr(key).upper()
                if letter not in ['B', 'D', 'H', 'Q', 'S', 'T']:
                    collect_data(letter, landmarks)
        if key == ord('m'):
            clf = train_svm_model()
            if clf is not None:
                feedback_message = "SVM Model Trained!"
                feedback_start_time = current_time
                speech_queue.put("SVM Model Trained")
                print("SVM model trained and loaded.")

# Cleanup
speech_queue.put("STOP")
speech_thread.join()
cap.release()
cv2.destroyAllWindows()
engine.stop()