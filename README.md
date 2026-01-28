🚀 Real-Time Object Detection & Tracking using YOLOv8

A real-time object detection and tracking system built using YOLOv8, capable of identifying people and common objects across both local video files and live camera streams.
The system performs fast inference, visualizes bounding boxes, tracks objects across frames, and displays live analytics through an interactive Gradio-based UI.

📌 Features

🔍 Real-time object detection using YOLOv8

🎯 Multi-object tracking across video frames

📹 Supports local video files and live camera streams

🧮 Object counting with dynamic updates

🖼️ Bounding box visualization with class labels & confidence scores

📊 Live analytics dashboard (detection mode, counts, stream status)

🧑‍💻 Interactive Gradio UI for easy visualization and control

🧠 Tech Stack

Model: YOLOv8 (Ultralytics)

Language: Python

Libraries & Tools:

OpenCV

PyTorch

Ultralytics YOLOv8

Gradio

NumPy

🏗️ System Architecture (High Level)

Video input (local file or live stream) is captured frame-by-frame

Frames are passed to the YOLOv8 model for detection

Detection results are forwarded to the tracking algorithm

Bounding boxes, labels, and object IDs are drawn on frames

Object counts and stream mode are computed in real time

Output is rendered live through a Gradio-based UI

🎥 Detection Modes

Local Video Mode

Automatically loads and processes video files

Suitable for offline analysis and testing

Live Stream Mode

Processes real-time camera feeds

Optimized for smooth frame handling

🖥️ User Interface (Gradio)

The Gradio UI allows users to:

View real-time detection output

Switch between local and live stream modes

Monitor object counts dynamically

Visualize bounding boxes and tracking IDs in real time

This removes the need for manual OpenCV windows and provides a clean, browser-based interface.

📂 Project Structure
├── models/
│   └── yolov8.pt
├── videos/
│   └── sample_video.mp4
├── utils/
│   ├── tracker.py
│   └── video_utils.py
├── app.py
├── requirements.txt
└── README.md

⚙️ Installation

Clone the repository:

git clone https://github.com/your-username/real-time-object-detection-yolov8.git
cd real-time-object-detection-yolov8


Install dependencies:

pip install -r requirements.txt


Run the application:

python app.py

🚀 Output

Real-time video stream with:

Bounding boxes

Object labels

Unique tracking IDs

Live object counts

Interactive browser-based dashboard via Gradio

🎯 Use Cases

Surveillance & monitoring systems

Crowd analysis

Smart traffic & object counting

Real-time video analytics applications

Computer vision research & demos

🌱 Future Enhancements

Cross-camera re-identification (ReID)

Performance optimization using GPU batching

Advanced analytics (heatmaps, dwell time)

Exporting detection summaries

🧑‍💻 Author

Gunari Nisitha Sri
Computer Science & Design
Passionate about Computer Vision, AI/ML, and building systems that work in real time.
