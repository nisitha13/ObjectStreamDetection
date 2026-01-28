from ultralytics import YOLO
import cv2
import gradio as gr
from collections import defaultdict, deque

# ================= CONFIG =================
STREAM_URL = "http://10.33.128.240:8080/video"
MODEL_PATH = "yolov8n.pt"

CONF_THRES = 0.3
IOU_THRES = 0.5

MIN_FRAMES_FOR_VALID_OBJECT = 4
HISTORY_LENGTH = 12 #dequeue length for object memory
STATIC_MOVEMENT_THRESHOLD = 8

DETECT_EVERY_N_FRAMES = 3 # detects every 3rd frame, yolo runs on every 3rd frame
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 600
# ==========================================

IS_LIVE_STREAM = STREAM_URL.startswith("http") or STREAM_URL.startswith("rtsp")
MODE_TEXT = "LIVE" if IS_LIVE_STREAM else "RECORDED"

# ================= LOAD MODEL =================
model = YOLO(MODEL_PATH)
model.conf = CONF_THRES
model.iou = IOU_THRES
# =============================================

# ================= OBJECT MEMORY =================
class ObjectMemory:
    def __init__(self):
        self.class_history = deque(maxlen=HISTORY_LENGTH)
        self.box_history = deque(maxlen=HISTORY_LENGTH)

    def update(self, cls, box):
        self.class_history.append(cls)
        self.box_history.append(box)

object_db = defaultdict(ObjectMemory)

# function use a fixed-size memory that automatically 
# removes old data and keeps only the latest frames 
# for each object.

# =================================================

# ================= INFERENCE =================
def infer_object(obj_mem):
    if len(obj_mem.class_history) < MIN_FRAMES_FOR_VALID_OBJECT:
        return None

    final_class = max(
        set(obj_mem.class_history),
        key=obj_mem.class_history.count
    )

    if len(obj_mem.box_history) >= 2:
        x1, y1, _, _ = obj_mem.box_history[0]
        x1b, y1b, _, _ = obj_mem.box_history[-1]
        if abs(x1 - x1b) + abs(y1 - y1b) < STATIC_MOVEMENT_THRESHOLD:
            return f"{final_class} (static)"

    return final_class
# =============================================

# ================= VIDEO GENERATOR =================
def stream_generator():

    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        yield None, "0", "0", MODE_TEXT, ""
        return

    frame_count = 0
    last_results = None

    while True:

        if IS_LIVE_STREAM:
            for _ in range(6):
                cap.grab()
            ret, frame = cap.retrieve()
        else:
            ret, frame = cap.read()

        if not ret or frame is None:
            break

        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        frame_count += 1

        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            last_results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

        people_count = 0
        object_count = 0
        logs = []

        if last_results and last_results[0].boxes.id is not None:
            boxes = last_results[0].boxes.xyxy.cpu().numpy()
            classes = last_results[0].boxes.cls.cpu().numpy()
            ids = last_results[0].boxes.id.cpu().numpy()

            for box, cls, track_id in zip(boxes, classes, ids):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(cls)]

                object_count += 1
                if class_name == "person":
                    people_count += 1

                obj = object_db[int(track_id)]
                obj.update(class_name, (x1, y1, x2, y2))

                inferred = infer_object(obj)
                if inferred:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        inferred,
                        (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )
                    logs.append(f"ID {int(track_id)} → {inferred}")

        yield (
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            str(people_count),
            str(object_count),
            MODE_TEXT,
            "\n".join(logs[-6:])
        )
# ==================================================

# ================= DASHBOARD =================
with gr.Blocks(
    css="""
    html, body, .gradio-container {
        background-color: #0a0e14 !important;
        color: #e5e7eb !important;
    }
    h1 {
        text-align: center;
        font-weight: 600;
        color: #e5e7eb;
    }
    .block, .panel {
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 10px;
    }
    label {
        color: #9ca3af !important;
    }
    textarea, input {
        background-color: #020617 !important;
        color: #e5e7eb !important;
        border: 1px solid #1f2937 !important;
    }
    """
) as demo:

    gr.Markdown("<h1>🛡 Real-Time Object Detection & Inference Dashboard</h1>")

    with gr.Row():
        video = gr.Image(label="Video Stream", streaming=True)

        with gr.Column(scale=0.55):
            people_box = gr.Textbox(label="👤 People", interactive=False)
            object_box = gr.Textbox(label="📦 Objects", interactive=False)
            mode_box = gr.Textbox(label="🟢 Mode", interactive=False)
            logs = gr.Textbox(label="🧠 Inference Logs", lines=8, interactive=False)

    demo.load(
        fn=stream_generator,
        inputs=None,
        outputs=[
            video,
            people_box,
            object_box,
            mode_box,
            logs
        ]
    )

demo.launch()
# ==================================================
