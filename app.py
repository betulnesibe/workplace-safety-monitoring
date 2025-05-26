import streamlit as st
import cv2
import numpy as np
import torch
from pathlib import Path
import tempfile
import os
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import time
import subprocess
import base64

# Constants
MODEL_PATH = "runs/detect/crossval/fold_2_run/weights/best.pt"

CLASS_NAMES = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 
               'Person', 'Safety Cone', 'Safety Vest', 'Machinery', 'Vehicle']

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []

def load_model():
    """Load the model"""
    if st.session_state.model is None:
        with st.spinner('Loading model... This might take a few minutes...'):
            st.session_state.model = YOLO(MODEL_PATH)
    return st.session_state.model

def process_image(image, model, target_size=(640, 640)):
    """Process image through the model with optional resizing"""
    # Resize image for faster processing
    height, width = image.shape[:2]
    if target_size:
        image = cv2.resize(image, target_size)
    
    results = model(image)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    labels = results.boxes.cls.cpu().numpy().astype(int)

    # Convert boxes back to original scale
    boxes[:, [0, 2]] *= width / target_size[0]
    boxes[:, [1, 3]] *= height / target_size[1]

    return boxes, scores, labels

def get_class_color(label):
    """Get color for each class with related classes using similar colors"""
    # Define colors in BGR format (OpenCV uses BGR)
    colors = {
        # PPE Compliance - Green shades for compliance, Red for violations
        0: (0, 255, 0),      # Hardhat - Green
        2: (0, 0, 255),      # NO-Hardhat - Red
        
        # Mask Compliance - Blue shades for compliance, Red for violations
        1: (255, 0, 0),      # Mask - Blue
        3: (0, 0, 255),      # NO-Mask - Red
        
        # Safety Vest Compliance - Yellow shades for compliance, Red for violations
        7: (0, 255, 255),    # Safety Vest - Yellow
        4: (0, 0, 255),      # NO-Safety Vest - Red
        
        # Other Safety Elements - Different colors
        5: (255, 255, 255),  # Person - White
        6: (0, 165, 255),    # Safety Cone - Orange
        8: (255, 0, 255),    # Machinery - Magenta
        9: (128, 0, 128)     # Vehicle - Purple
    }
    return colors.get(label, (255, 255, 255))  # Default to white if label not found

def is_violation_class(label):
    """Check if the class is a violation class (has 'NO-' prefix)"""
    return label in [2, 3, 4]  # NO-Hardhat, NO-Mask, NO-Safety Vest

def draw_detections(image, boxes, scores, labels, class_filter=None):
    """Draw detection boxes on image with improved visual display and class filtering"""
    for box, score, label in zip(boxes, scores, labels):
        label_int = int(label)
        
        # Skip if class is filtered out
        if class_filter is not None and label_int not in class_filter:
            continue
            
        x1, y1, x2, y2 = map(int, box)
        color = get_class_color(label_int)
        
        # Draw main rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        
        # Add red border for violation classes
        if is_violation_class(label_int):
            cv2.rectangle(image, (x1-1, y1-1), (x2+1, y2+1), (0, 0, 255), 1)
        
        # Create label text with class name and confidence
        label_text = f'{CLASS_NAMES[label_int]} {score:.2f}'
        
        # Get text size for background rectangle
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )
        
        # Add padding to text background
        padding = 2
        text_bg_x1 = x1
        text_bg_y1 = y1 - text_height - 2 * padding
        text_bg_x2 = x1 + text_width + 2 * padding
        text_bg_y2 = y1
        
        # Draw background rectangle for text with slight transparency
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (text_bg_x1, text_bg_y1),
            (text_bg_x2, text_bg_y2),
            color,
            -1
        )
        # Add slight transparency to the background
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw text with improved visibility
        cv2.putText(
            image,
            label_text,
            (x1 + padding, y1 - padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,  # Slightly smaller font
            (255, 255, 255),  # White text
            1,
            cv2.LINE_AA  # Anti-aliased text
        )
    return image

def create_legend():
    """Create a professional legend for the detection colors"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Detection Legend")
    
    # PPE Compliance
    st.sidebar.markdown("**PPE Compliance**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.markdown("🟢 Hardhat")
        st.markdown("🔵 Mask")
        st.markdown("🟡 Safety Vest")
    with col2:
        st.markdown("🔴 NO-Hardhat")
        st.markdown("🔴 NO-Mask")
        st.markdown("🔴 NO-Safety Vest")
    
    # Other Safety Elements
    st.sidebar.markdown("**Safety Elements**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.markdown("⚪ Person")
        st.markdown("🟠 Safety Cone")
    with col2:
        st.markdown("🟣 Machinery")
        st.markdown("🟣 Vehicle")

def analyze_safety_violations(labels, scores):
    """Analyze safety violations based on detected objects"""
    violations = []
    if any(label == 2 for label in labels):  # NO-Hardhat
        violations.append("⚠️ Hardhat violation detected")
    if any(label == 3 for label in labels):  # NO-Mask
        violations.append("⚠️ Mask violation detected")
    if any(label == 4 for label in labels):  # NO-Safety Vest
        violations.append("⚠️ Safety vest violation detected")
    return violations

def compress_video(input_path, output_path, target_size_mb=10):
    """Compress video to target size while maintaining quality"""
    # Get video properties
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Calculate target bitrate (bits per second)
    target_size_bits = target_size_mb * 8 * 1024 * 1024
    duration_seconds = frame_count / fps
    target_bitrate = int(target_size_bits / duration_seconds)

    # FFmpeg command for compression with optimized settings
    command = [
        'ffmpeg', '-i', input_path,
        '-c:v', 'libx264',
        '-b:v', f'{target_bitrate}',
        '-maxrate', f'{target_bitrate * 1.5}',
        '-bufsize', f'{target_bitrate * 2}',
        '-preset', 'faster',  # Changed from 'medium' to 'faster'
        '-tune', 'fastdecode',  # Added for faster decoding
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-threads', '0',  # Use all available CPU threads
        output_path
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error compressing video: {e.stderr.decode()}")
        return False

def analyze_video_violations(labels_list, scores_list, fps):
    """Analyze violations in video frames with timestamps and remove duplicates"""
    violations = {
        'NO-Hardhat': {'count': 0, 'timestamps': []},
        'NO-Mask': {'count': 0, 'timestamps': []},
        'NO-Safety Vest': {'count': 0, 'timestamps': []}
    }
    
    # Time window to consider violations as duplicates (in seconds)
    duplicate_window = 2.0  # 2 seconds window
    
    for frame_idx, (labels, scores) in enumerate(zip(labels_list, scores_list)):
        timestamp = frame_idx / fps
        for label, score in zip(labels, scores):
            if score > confidence_threshold:
                if label == 2:  # NO-Hardhat
                    violations['NO-Hardhat']['count'] += 1
                    # Only add timestamp if it's not within duplicate window of previous violation
                    if not violations['NO-Hardhat']['timestamps'] or \
                       timestamp - violations['NO-Hardhat']['timestamps'][-1] > duplicate_window:
                        violations['NO-Hardhat']['timestamps'].append(timestamp)
                elif label == 3:  # NO-Mask
                    violations['NO-Mask']['count'] += 1
                    if not violations['NO-Mask']['timestamps'] or \
                       timestamp - violations['NO-Mask']['timestamps'][-1] > duplicate_window:
                        violations['NO-Mask']['timestamps'].append(timestamp)
                elif label == 4:  # NO-Safety Vest
                    violations['NO-Safety Vest']['count'] += 1
                    if not violations['NO-Safety Vest']['timestamps'] or \
                       timestamp - violations['NO-Safety Vest']['timestamps'][-1] > duplicate_window:
                        violations['NO-Safety Vest']['timestamps'].append(timestamp)
    
    return violations

def format_timestamp(seconds):
    """Format seconds into MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def highlight_violation_frame(frame, violations):
    """Add visual indicators for violations in the frame"""
    height, width = frame.shape[:2]
    
    # Add semi-transparent overlay for violation frames
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), 50)  # Red border
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)  # Blend overlay
    
    # Add violation indicators
    y_offset = 30
    for violation_type, count in violations.items():
        if count > 0:
            text = f"{violation_type}: {count}"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
    
    return frame

def get_video_display_size(width, height, max_width=800, max_height=600):
    """Calculate the display size while maintaining aspect ratio"""
    aspect_ratio = width / height
    
    if width > max_width:
        width = max_width
        height = int(width / aspect_ratio)
    
    if height > max_height:
        height = max_height
        width = int(height * aspect_ratio)
    
    return width, height

# Streamlit UI
st.title("🏗️ Workplace Safety Monitoring System")
st.write("Upload an image, video, or use webcam to detect safety violations")

# Add legend to sidebar
create_legend()

# Sidebar
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Class filtering
st.sidebar.markdown("---")
st.sidebar.subheader("Detection Filters")

# Violation Filter
violation_col1, violation_col2 = st.sidebar.columns([3, 1])
with violation_col1:
    st.markdown("**Safety Violations**")
with violation_col2:
    show_violations = st.checkbox("", value=False, key="violations_toggle", 
                                help="Show only safety violations (NO-Hardhat, NO-Mask, NO-Safety Vest)")

# PPE Compliance Filtering
st.sidebar.markdown("---")
ppe_col1, ppe_col2 = st.sidebar.columns([3, 1])
with ppe_col1:
    st.markdown("**PPE Compliance**")
with ppe_col2:
    show_ppe = st.checkbox("", value=True, key="ppe_toggle", 
                          help="Toggle all PPE detection (both compliance and violations)",
                          disabled=show_violations)

if show_ppe and not show_violations:
    # Create indented container for PPE filters
    with st.sidebar.container():
        # Hardhat Detection
        hardhat_col1, hardhat_col2 = st.columns([3, 1])
        with hardhat_col1:
            st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;Hardhat Detection")
        with hardhat_col2:
            show_hardhat = st.checkbox("", value=True, key="hardhat", 
                                     help="Toggle hardhat detection (both wearing and not wearing)")
        
        # Mask Detection
        mask_col1, mask_col2 = st.columns([3, 1])
        with mask_col1:
            st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;Mask Detection")
        with mask_col2:
            show_mask = st.checkbox("", value=True, key="mask", 
                                  help="Toggle mask detection (both wearing and not wearing)")
        
        # Safety Vest Detection
        vest_col1, vest_col2 = st.columns([3, 1])
        with vest_col1:
            st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;Safety Vest Detection")
        with vest_col2:
            show_vest = st.checkbox("", value=True, key="vest", 
                                  help="Toggle safety vest detection (both wearing and not wearing)")

# Safety Elements Filtering
st.sidebar.markdown("---")
safety_col1, safety_col2 = st.sidebar.columns([3, 1])
with safety_col1:
    st.markdown("**Safety Elements**")
with safety_col2:
    show_safety = st.checkbox("", value=True, key="safety_toggle", 
                            help="Toggle detection of safety elements and equipment",
                            disabled=show_violations)

if show_safety and not show_violations:
    # Create indented container for safety filters
    with st.sidebar.container():
        # Person Detection
        person_col1, person_col2 = st.columns([3, 1])
        with person_col1:
            st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;Person")
        with person_col2:
            show_person = st.checkbox("", value=True, key="person", 
                                    help="Toggle person detection")
        
        # Safety Cone Detection
        cone_col1, cone_col2 = st.columns([3, 1])
        with cone_col1:
            st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;Safety Cone")
        with cone_col2:
            show_cone = st.checkbox("", value=True, key="cone", 
                                  help="Toggle safety cone detection")
        
        # Machinery Detection
        machinery_col1, machinery_col2 = st.columns([3, 1])
        with machinery_col1:
            st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;Machinery")
        with machinery_col2:
            show_machinery = st.checkbox("", value=True, key="machinery", 
                                       help="Toggle machinery detection")
        
        # Vehicle Detection
        vehicle_col1, vehicle_col2 = st.columns([3, 1])
        with vehicle_col1:
            st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;Vehicle")
        with vehicle_col2:
            show_vehicle = st.checkbox("", value=True, key="vehicle", 
                                     help="Toggle vehicle detection")

# Create class filter list based on checkbox states
class_filter = []

# Handle violation-only filter
if show_violations:
    class_filter = [2, 3, 4]  # Only NO-Hardhat, NO-Mask, NO-Safety Vest
else:
    if show_ppe:
        if show_hardhat:
            class_filter.extend([0, 2])  # Hardhat and NO-Hardhat
        if show_mask:
            class_filter.extend([1, 3])  # Mask and NO-Mask
        if show_vest:
            class_filter.extend([7, 4])  # Safety Vest and NO-Safety Vest

    if show_safety:
        if show_person:
            class_filter.append(5)  # Person
        if show_cone:
            class_filter.append(6)  # Safety Cone
        if show_machinery:
            class_filter.append(8)  # Machinery
        if show_vehicle:
            class_filter.append(9)  # Vehicle

# Main content
tab1, tab2, tab3 = st.tabs(["Image Upload", "Video Upload", "Webcam"])

with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Load model
        model = load_model()
        
        # Read and process image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Process image
        boxes, scores, labels = process_image(image, model)
        
        # Filter by confidence
        mask = scores > confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # Draw detections with class filtering
        result_image = draw_detections(image.copy(), boxes, scores, labels, class_filter)
        
        # Convert BGR to RGB for display
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        # Display results
        st.image(result_image, caption="Detection Results", use_container_width=True)
        
        # Analyze violations
        violations = analyze_safety_violations(labels, scores)
        if violations:
            st.warning("Safety Violations Detected:")
            for violation in violations:
                st.write(violation)
        else:
            st.success("✅ No safety violations detected!")

with tab2:
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi"])
    if uploaded_video is not None:
        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        # Load model
        model = load_model()
        
        # Process video
        cap = cv2.VideoCapture(tfile.name)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate display size
        display_width, display_height = get_video_display_size(width, height)
        
        # Create video writer with H.264 codec
        temp_output = "temp_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        out = cv2.VideoWriter(temp_output, fourcc, fps, (display_width, display_height))
        
        progress_bar = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create a fixed-size container for the video
        video_container = st.container()
        with video_container:
            # Add custom CSS for the video container
            st.markdown("""
                <style>
                .video-container {
                    width: 800px;
                    height: 600px;
                    margin: 0 auto;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    background-color: #000;
                }
                .video-container video {
                    max-width: 100%;
                    max-height: 100%;
                    object-fit: contain;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Create a placeholder for the video
            video_placeholder = st.empty()
        
        # Lists to store detection results for analysis
        all_labels = []
        all_scores = []
        
        # Process every nth frame (adjust this value based on performance needs)
        frame_skip = 5  # Process every 5th frame
        
        # Add info about frame skipping
        st.info("Note: Processing every 5th frame for faster analysis. This may miss very brief violations but provides good coverage for most safety checks.")
        
        for i in range(0, frame_count, frame_skip):
            # Skip frames
            for _ in range(frame_skip - 1):
                cap.grab()
            
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame to display size
            frame = cv2.resize(frame, (display_width, display_height))
                
            # Process frame at lower resolution
            boxes, scores, labels = process_image(frame, model, target_size=(640, 640))
            
            # Store results for analysis
            all_labels.append(labels)
            all_scores.append(scores)
            
            # Filter by confidence
            mask = scores > confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            # Count violations in current frame
            frame_violations = {
                'NO-Hardhat': sum(1 for l in labels if l == 2),
                'NO-Mask': sum(1 for l in labels if l == 3),
                'NO-Safety Vest': sum(1 for l in labels if l == 4)
            }
            
            # Draw detections with class filtering
            result_frame = draw_detections(frame, boxes, scores, labels, class_filter)
            
            # Highlight frame if it contains violations
            if any(count > 0 for count in frame_violations.values()):
                result_frame = highlight_violation_frame(result_frame, frame_violations)
            
            # Write frame
            out.write(result_frame)
            
            # Update progress
            progress_bar.progress((i + frame_skip) / frame_count)
        
        cap.release()
        out.release()
        
        # Clean up the temporary input file
        os.unlink(tfile.name)
        
        # Compress the video with optimized settings
        output_path = "output_video.mp4"
        with st.spinner("Compressing video..."):
            if compress_video(temp_output, output_path):
                os.unlink(temp_output)  # Remove temporary output
            else:
                output_path = temp_output  # Use uncompressed version if compression fails
        
        # Analyze violations with timestamps (adjust timestamps for frame skipping)
        violations = analyze_video_violations(all_labels, all_scores, fps/frame_skip)
        
        # Display the processed video in the fixed container
        try:
            with open(output_path, 'rb') as video_file:
                video_bytes = video_file.read()
                with video_container:
                    video_placeholder.markdown(f"""
                        <div class="video-container">
                            <video controls>
                                <source src="data:video/mp4;base64,{base64.b64encode(video_bytes).decode()}" type="video/mp4">
                                Your browser does not support the video tag.
                            </video>
                        </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying video: {str(e)}")
            st.info("If the video doesn't play, you can download it using the button below.")
            with open(output_path, 'rb') as video_file:
                st.download_button(
                    label="Download processed video",
                    data=video_file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
        
        # Display violation analysis
        st.markdown("---")
        st.subheader("Video Analysis")
        
        # Create columns for violation metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Hardhat Violations", violations['NO-Hardhat']['count'])
        with col2:
            st.metric("Mask Violations", violations['NO-Mask']['count'])
        with col3:
            st.metric("Safety Vest Violations", violations['NO-Safety Vest']['count'])
        
        # Display violation timeline
        st.markdown("### Violation Timeline")
        timeline_data = {
            'Frame': list(range(len(all_labels))),
            'NO-Hardhat': [sum(1 for l, s in zip(labels, scores) if l == 2 and s > confidence_threshold) 
                          for labels, scores in zip(all_labels, all_scores)],
            'NO-Mask': [sum(1 for l, s in zip(labels, scores) if l == 3 and s > confidence_threshold) 
                       for labels, scores in zip(all_labels, all_scores)],
            'NO-Safety Vest': [sum(1 for l, s in zip(labels, scores) if l == 4 and s > confidence_threshold) 
                             for labels, scores in zip(all_labels, all_scores)]
        }
        st.line_chart(timeline_data)
        
        # Display violation timestamps
        st.markdown("### Violation Timestamps")
        for violation_type, data in violations.items():
            if data['count'] > 0:
                st.markdown(f"**{violation_type}**")
                timestamps = [format_timestamp(ts) for ts in data['timestamps']]
                st.write(", ".join(timestamps))
        
        # Clean up the output file
        os.unlink(output_path)

with tab3:
    st.write("Real-time Safety Monitoring")
    
    # Initialize session state for webcam
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    
    # Create columns for controls
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("Start Webcam" if not st.session_state.webcam_active else "Stop Webcam"):
            st.session_state.webcam_active = not st.session_state.webcam_active
            st.rerun()
    
    with col2:
        if st.session_state.webcam_active:
            st.info("Webcam is active. Press 'q' to stop the webcam feed.")
    
    # Create a fixed-size container for the webcam feed
    webcam_container = st.container()
    with webcam_container:
        st.markdown("""
            <style>
            .webcam-container {
                width: 800px;
                height: 600px;
                margin: 0 auto;
                display: flex;
                justify-content: center;
                align-items: center;
                background-color: #000;
            }
            .webcam-container img {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Create a placeholder for the webcam feed
        webcam_placeholder = st.empty()
    
    if st.session_state.webcam_active:
        # Load model
        model = load_model()
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        try:
            while st.session_state.webcam_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam")
                    break
                
                # Process frame
                boxes, scores, labels = process_image(frame, model, target_size=(640, 640))
                
                # Filter by confidence
                mask = scores > confidence_threshold
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]
                
                # Count violations in current frame
                frame_violations = {
                    'NO-Hardhat': sum(1 for l in labels if l == 2),
                    'NO-Mask': sum(1 for l in labels if l == 3),
                    'NO-Safety Vest': sum(1 for l in labels if l == 4)
                }
                
                # Draw detections with class filtering
                result_frame = draw_detections(frame, boxes, scores, labels, class_filter)
                
                # Highlight frame if it contains violations
                if any(count > 0 for count in frame_violations.values()):
                    result_frame = highlight_violation_frame(result_frame, frame_violations)
                
                # Convert BGR to RGB for display
                result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                
                # Display the frame
                with webcam_container:
                    webcam_placeholder.image(result_frame, channels="RGB", use_container_width=True)
                
                # Check for 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    st.session_state.webcam_active = False
                    st.rerun()
                    break
                
        except Exception as e:
            st.error(f"Error accessing webcam: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    else:
        # Display placeholder when webcam is inactive
        with webcam_container:
            webcam_placeholder.markdown("""
                <div class="webcam-container">
                    <p style="color: white; text-align: center;">Click 'Start Webcam' to begin real-time safety monitoring</p>
                </div>
            """, unsafe_allow_html=True) 