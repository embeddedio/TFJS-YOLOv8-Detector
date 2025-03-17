# YOLOv8 Live Detection App

## Introduction

The **YOLOv8 Live Detection App** is a browser-based application that leverages TensorFlow\.js to perform real-time object detection using the YOLOv8 model. This app allows users to upload images or videos, or use their webcam to detect objects in real-time. It is designed to be user-friendly and can be easily integrated into web applications.

## Features

- **Real-time Object Detection**: Detect objects in real-time from images, videos, or webcam feeds.
- **Multiple Input Sources**: Supports image uploads, video uploads, and webcam streams.
- **Confidence Threshold**: Adjustable confidence threshold to filter out low-confidence detections.
- **Responsive Design**: Fully responsive layout that works seamlessly on both desktop and mobile devices.

## Technologies Used

- **TensorFlow\.js**: A JavaScript library for training and deploying machine learning models in the browser and on Node.js.
- **HTML, CSS, JavaScript**: Used to build the front-end interface.

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/yolov8-live-detection-app.git
cd yolov8-live-detection-app
```

### Load the Model

Make sure you have the YOLOv8 model files (`model/model.json` and the binary models files available in the `model` directory. These files can be downloaded from the official YOLOv8 repository or generated using the YOLOv8 training pipeline.

## Running the Application

To run the application, you need to serve it using an HTTP server. You can use one of the following methods:

### Using Node.js Live Server

Install the `live-server` module globally:

```bash
npm install -g live-server
```

Run the server in the project directory:

```bash
live-server
```

### Using Python HTTP Server

For Python 3:

```bash
python -m http.server 8000
```

For Python 2:

```bash
python -m SimpleHTTPServer 8000
```

Then, open `http://localhost:8000` in your browser.

## Usage

### Opening Images

1. Click the **"Open Image"** button.
2. Select an image from your local file system.
3. The selected image will be displayed with detected objects highlighted.

### Opening Videos

1. Click the **"Open Video"** button.
2. Select a video from your local file system.
3. The selected video will start playing with detected objects highlighted.

### Using Webcam

1. Click the **"Open Webcam"** button.
2. Grant permission to access your webcam.
3. The webcam feed will start displaying with detected objects highlighted.

### Adjusting Confidence Threshold

Use the slider below the buttons to adjust the confidence threshold. This controls the sensitivity of the object detection algorithm.

## How It Works

1. **Model Loading**: The application loads the pre-trained YOLOv8 model using TensorFlow\.js.
2. **Preprocessing**: Input images or frames from videos/webcams are preprocessed to fit the model's input shape.
3. **Detection**: The model predicts bounding boxes and class probabilities for each object in the input.
4. **Non-Maximum Suppression (NMS)**: NMS is applied to remove overlapping bounding boxes and retain the most confident detections.
5. **Rendering**: Detected objects are drawn on a canvas overlaying the input image or video frame.

## Converting YOLOv8 PyTorch Model to TensorFlow\.js

If you have a YOLOv8 model trained in PyTorch (`.pt`), you can convert it to TensorFlow\.js format using Python. Here’s how:

```python
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolo11n.pt")  # Replace with your .pt file path

# Export to TensorFlow.js
model.export(format="tfjs")
```

This will generate the necessary TensorFlow\.js model files (`model.json` and binary shard files) that you can use in this application.

## Contributing

Contributions are welcome! Please read the **CONTRIBUTING.md** file for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the **MIT License** - see the `LICENSE` file for details.

## Acknowledgments

- **YOLOv8**: The YOLOv8 model used in this application.
- **TensorFlow\.js**: The JavaScript library for training and deploying machine learning models in the browser.

## Contact

If you have any questions or need assistance, feel free to contact us at [[ehyoussefl@gmail.com](mailto:ehyoussef@gmail.com)].

