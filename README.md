# Workplace Safety Monitoring System

A real-time workplace safety monitoring system that uses AI to detect safety violations, including PPE compliance (hardhats, masks, safety vests) and other safety elements in the workplace.

## Features

- **Real-time Safety Monitoring**
  - Webcam-based live monitoring
  - Instant violation detection
  - Visual highlighting of safety violations

- **Image Analysis**
  - Upload and analyze single images
  - Detect safety violations
  - Visual feedback with bounding boxes

- **Video Analysis**
  - Process uploaded videos
  - Frame-by-frame violation detection
  - Violation timeline and statistics
  - Timestamp-based violation tracking

- **Detection Capabilities**
  - PPE Compliance:
    - Hardhat detection (wearing/not wearing)
    - Mask detection (wearing/not wearing)
    - Safety vest detection (wearing/not wearing)
  - Safety Elements:
    - Person detection
    - Safety cone detection
    - Machinery detection
    - Vehicle detection

- **Customizable Settings**
  - Adjustable confidence threshold
  - Class-specific filtering
  - Grouped detection categories

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for better performance)
- Webcam (for real-time monitoring)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd workplace-safety-monitoring
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the YOLO model weights:
   - Place the model weights in the following directory structure:
   ```
   runs/
   └── detect/
       └── crossval/
           └── fold_2_run/
               └── weights/
                   └── best.pt
   ```

## Running the Application

1. Activate the virtual environment (if not already activated):
```bash
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Usage

### Image Analysis
1. Select the "Image Upload" tab
2. Upload an image file (supported formats: jpg, jpeg, png)
3. View the detection results and any safety violations

### Video Analysis
1. Select the "Video Upload" tab
2. Upload a video file (supported formats: mp4, avi)
3. Wait for the video processing to complete
4. View the processed video with detections and violation analysis

### Real-time Monitoring
1. Select the "Webcam" tab
2. Click "Start Webcam" to begin real-time monitoring
3. Press 'q' to stop the webcam feed

## Customization

### Detection Filters
- Use the sidebar to adjust the confidence threshold
- Toggle specific detection categories:
  - PPE Compliance (Hardhat, Mask, Safety Vest)
  - Safety Elements (Person, Safety Cone, Machinery, Vehicle)
- Show only violations or all detections

### Performance Optimization
- The application automatically optimizes video processing by:
  - Processing every 5th frame for faster analysis
  - Reducing resolution for better performance
  - Compressing output videos

## Troubleshooting

1. **Webcam Access Issues**
   - Ensure your webcam is properly connected
   - Check if other applications are using the webcam
   - Verify webcam permissions in your system settings

2. **Model Loading Issues**
   - Verify that the model weights are in the correct directory
   - Check if CUDA is properly installed (if using GPU)
   - Ensure sufficient disk space is available

3. **Performance Issues**
   - Reduce the input video resolution
   - Increase the frame skip value
   - Close other resource-intensive applications

## Acknowledgements

- YOLO model architecture and implementation
- Streamlit for the web application framework
- OpenCV for image and video processing 
