"""
YOLO Object Detection Web Application
===================================

This Streamlit application provides a web interface for performing object detection using the YOLO (You Only Look Once)
model with TensorRT optimization. It supports both image and video processing with GPU acceleration.

Features:
- Image upload and processing
- Video upload and batch processing
- Real-time GPU memory monitoring
- Adjustable confidence threshold
- Download capability for processed media
- Progress tracking for video processing

Requirements:
- CUDA-capable GPU
- TensorRT
- Streamlit
- Ultralytics YOLO
- OpenCV
- PIL
- FFmpeg

Author: [Your Name]
Date: [Current Date]
"""

import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import os
import torch
import tempfile
import subprocess
from pathlib import Path
import queue
import threading
from datetime import datetime
import shutil

# Check for GPU availability
if not torch.cuda.is_available():
    st.error("TensorRT engine requires CUDA GPU. No GPU found!")
    st.stop()

# Configure GPU settings
DEVICE = 'cuda'
torch.cuda.set_device(0)
st.sidebar.success(f"Using GPU: {torch.cuda.get_device_name(0)}")

st.title("YOLO Object Detection")

# Configuration constants
MODEL_PATH = "D:/1Work/Projects/Object Detection/Urban Object Detection/final train results/weights/best.engine"
INPUT_SIZE = (1280, 1280)

def process_image(image: Image.Image) -> Image.Image:
    """
    Resize and pad an input image to match the model's expected input size.
    
    Args:
        image (PIL.Image): Input image to be processed
        
    Returns:
        PIL.Image: Processed image with consistent dimensions and padding
        
    The function performs the following steps:
    1. Calculates the aspect ratio preserving resize factor
    2. Resizes the image using LANCZOS resampling
    3. Adds padding to maintain constant input size
    """
    target_size = INPUT_SIZE
    img = image.copy()
    ratio = min(target_size[0] / img.size[0], target_size[1] / img.size[1])
    new_size = tuple(int(dim * ratio) for dim in img.size)
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Create new padded image with gray background (114, 114, 114 is YOLO's default padding color)
    new_img = Image.new("RGB", target_size, (114, 114, 114))
    new_img.paste(img, ((target_size[0] - new_size[0]) // 2,
                       (target_size[1] - new_size[1]) // 2))
    return new_img

@st.cache_resource
def load_model(model_path: str) -> YOLO:
    """
    Load the YOLO model with TensorRT optimization.
    
    Args:
        model_path (str): Path to the TensorRT engine file
        
    Returns:
        YOLO: Loaded YOLO model object or None if loading fails
        
    The function is cached using Streamlit's cache_resource decorator to prevent
    reloading the model on every rerun.
    """
    try:
        if not os.path.exists(model_path):
            st.error(f"Engine file not found at: {model_path}")
            return None
        model = YOLO(model_path, task='detect')
        return model
    except Exception as e:
        st.error(f"Error loading TensorRT engine: {str(e)}")
        return None

class VideoProcessor:
    """
    Handles video processing with batch processing capabilities.
    
    This class implements a multi-threaded approach to video processing:
    - One thread extracts frames from the video
    - Another thread processes the frames using the YOLO model
    - Results are synchronized using thread-safe queues
    
    Attributes:
        source_path (str): Path to the input video file
        batch_size (int): Number of frames to process in each batch
        frame_queue (Queue): Queue for storing extracted video frames
        result_queue (Queue): Queue for storing processed frames
        processing_done (Event): Threading event to signal completion
        cap (cv2.VideoCapture): OpenCV video capture object
        fps (int): Frames per second of the input video
        total_frames (int): Total number of frames in the video
        width (int): Frame width
        height (int): Frame height
    """
    
    def __init__(self, source_path: str, batch_size: int = 4):
        """
        Initialize the VideoProcessor with video source and batch size.
        
        Args:
            source_path (str): Path to the input video file
            batch_size (int): Number of frames to process in each batch (default: 4)
        """
        self.source_path = source_path
        self.batch_size = batch_size
        self.frame_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue()
        self.processing_done = threading.Event()
        self.cap = cv2.VideoCapture(source_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def extract_frames(self):
        """
        Extract frames from the video and add them to the frame queue.
        
        This method runs in a separate thread and continuously reads frames
        from the video until all frames are extracted.
        """
        frame_count = 0
        while True:
            frames_batch = []
            for _ in range(self.batch_size):
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_count += 1
                frames_batch.append((frame_count, frame))
            
            if not frames_batch:
                break
                
            self.frame_queue.put(frames_batch)
        
        self.cap.release()
        self.processing_done.set()

    def process_frames(self, model: YOLO, conf_threshold: float):
        """
        Process frames using the YOLO model.
        
        Args:
            model (YOLO): Loaded YOLO model
            conf_threshold (float): Confidence threshold for detection
            
        This method runs in a separate thread and processes frames from the
        frame queue until all frames are processed.
        """
        while not (self.processing_done.is_set() and self.frame_queue.empty()):
            try:
                frames_batch = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            processed_batch = []
            for frame_idx, frame in frames_batch:
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                processed_frame = process_image(frame_pil)
                frame = cv2.cvtColor(np.array(processed_frame), cv2.COLOR_RGB2BGR)
                
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    device=DEVICE,
                    verbose=False,
                    imgsz=INPUT_SIZE
                )
                annotated_frame = results[0].plot()
                processed_batch.append((frame_idx, annotated_frame))
            
            self.result_queue.put(processed_batch)

def save_processed_video(processor: VideoProcessor, output_path: str, progress_bar):
    """
    Save processed video frames as a video file using FFmpeg.
    
    Args:
        processor (VideoProcessor): Video processor instance containing processed frames
        output_path (str): Path where the output video will be saved
        progress_bar (st.progress): Streamlit progress bar object
        
    This function:
    1. Creates a temporary directory for storing frame images
    2. Saves processed frames as sequential images
    3. Uses FFmpeg to combine images into a video
    4. Cleans up temporary files
    """
    temp_dir = Path(tempfile.mkdtemp())
    try:
        frame_files = []
        processed_frames = 0

        # Save frames as images
        while processed_frames < processor.total_frames:
            try:
                batch = processor.result_queue.get(timeout=30)
                for frame_idx, frame in batch:
                    frame_path = temp_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_files.append(frame_path)
                    processed_frames += 1
                    progress_bar.progress(processed_frames / processor.total_frames)
            except queue.Empty:
                break

        frame_files.sort()

        # FFmpeg command to create video
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(processor.fps),
            '-pattern_type', 'sequence',
            '-i', str(temp_dir / 'frame_%06d.jpg'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '23',
            output_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True)

    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir)

# Main application logic
model = load_model(MODEL_PATH)
if model is None:
    st.stop()

# UI Controls
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
fps_placeholder = st.sidebar.empty()

upload_type = st.radio("Upload type", ["Image", "Video"])

# Image processing section
if upload_type == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            processed_image = process_image(image)
            
            # Measure inference time
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            results = model.predict(
                source=processed_image,
                conf=conf_threshold,
                device=DEVICE,
                verbose=False,
                imgsz=INPUT_SIZE
            )
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)
            
            # Display results
            annotated_image = results[0].plot()
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_image, caption="Detected Image")
            st.sidebar.info(f"Inference time: {inference_time:.2f}ms")
            
            # Provide download option
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                Image.fromarray(annotated_image).save(tmp_file.name)
                with open(tmp_file.name, 'rb') as f:
                    st.download_button(
                        label="Download annotated image",
                        data=f.read(),
                        file_name="annotated_image.png",
                        mime="image/png"
                    )
                os.unlink(tmp_file.name)
            
            # Display detection results
            for r in results:
                for box in r.boxes:
                    st.write(f"Class: {model.names[int(box.cls)]} | Confidence: {box.conf.item():.2f}")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Video processing section
else:
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        try:
            # Handle video processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as input_tmp:
                input_tmp.write(uploaded_file.read())
                input_video_path = input_tmp.name

            output_video_path = str(Path(tempfile.mkdtemp()) / 'output.mp4')

            processor = VideoProcessor(input_video_path)
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Start processing threads
            extract_thread = threading.Thread(target=processor.extract_frames)
            extract_thread.start()

            process_thread = threading.Thread(
                target=processor.process_frames,
                args=(model, conf_threshold)
            )
            process_thread.start()

            # Process and save video
            status_text.text("Processing video...")
            save_processed_video(processor, output_video_path, progress_bar)

            # Wait for completion
            extract_thread.join()
            process_thread.join()

            status_text.text("Processing complete! You can now download the video.")
            
            # Provide download option
            with open(output_video_path, 'rb') as f:
                st.download_button(
                    label="Download annotated video",
                    data=f.read(),
                    file_name=f"annotated_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    mime="video/mp4"
                )

            # Cleanup temporary files
            os.unlink(input_video_path)
            os.unlink(output_video_path)
            os.rmdir(str(Path(output_video_path).parent))

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            # Cleanup in case of error
            for path in [input_video_path, output_video_path]:
                if os.path.exists(path):
                    os.unlink(path)
            if os.path.exists(str(Path(output_video_path).parent)):
                os.rmdir(str(Path(output_video_path).parent))

# Display GPU memory usage
gpu_memory = torch.cuda.memory_allocated(0) / 1024**2
st.sidebar.info(f"GPU Memory Used: {gpu_memory:.2f} MB")