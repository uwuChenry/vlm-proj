"""
Real-Time YOLO-World Object Detection Script

This script runs YOLO-World in real-time mode, continuously processing webcam frames
and displaying detection results live on screen. No need to capture individual images!

Features:
- Real-time object detection with live webcam feed
- Customizable object vocabulary (not limited to 80 COCO classes)
- GPU acceleration for smooth performance
- Interactive controls for adjusting settings
- Live FPS display and performance monitoring

Controls:
- 'q': Quit application
- 'c': Change detection classes
- 's': Save current frame with detections
- '+': Increase confidence threshold
- '-': Decrease confidence threshold
- 'r': Reset to default classes

Requirements:
- ultralytics (with YOLO-World support)
- OpenCV for webcam capture and display
- PyTorch with CUDA support for GPU acceleration
"""

import cv2
import os
import torch
import time
from datetime import datetime
from ultralytics import YOLOWorld
import numpy as np
from utils import plot_bounding_boxes_opencv

class RealTimeYOLOWorld:
    """Real-time YOLO-World open-vocabulary object detection"""
    
    def __init__(self, model_size='l', confidence_threshold=0.3, target_fps=30):
        """
        Initialize Real-time YOLO-World model
        
        Args:
            model_size: 's' (small), 'm' (medium), 'l' (large) - smaller is faster
            confidence_threshold: Minimum confidence for detections
            target_fps: Target FPS for processing (actual FPS depends on hardware)
        """
        self.confidence_threshold = confidence_threshold
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Force CUDA usage if available
        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"   Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   PyTorch Version: {torch.__version__}")
            
            # Set CUDA memory management for RTX 5090 using new PyTorch 2.9 API
            torch.backends.cudnn.benchmark = True
            # Use new TF32 API (PyTorch 2.9+)
            torch.backends.cudnn.conv.fp32_precision = 'tf32'
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
            
        else:
            self.device = 'cpu'
            print("WARNING: CUDA not available, using CPU (will be slower)")
        
        # Load YOLO-World model
        model_name = f'yolov8{model_size}-world.pt'
        print(f"Loading {model_name}...")
        
        try:
            self.model = YOLOWorld(model_name)
            
            # Force model to GPU with explicit device setting
            if self.device == 'cuda':
                self.model = self.model.to('cuda')
                print(" Model successfully moved to GPU")

                    
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to CPU...")
            self.device = 'cpu'
            self.model = YOLOWorld(model_name)
        
        # No default classes - YOLO-World is open vocabulary!
        # User must specify what they want to detect
        self.current_classes = []

    
    def set_classes(self, class_list):
        """
        Set custom classes for detection
        
        Args:
            class_list: List of class names to detect
        """
        print(f"Setting custom vocabulary with {len(class_list)} classes")
        print(f"Classes: {', '.join(class_list[:8])}{'...' if len(class_list) > 8 else ''}")
        
        self.model.set_classes(class_list)
        self.current_classes = class_list
    
    def detect_objects_fast(self, image):
        """
        Fast YOLO-World detection optimized for real-time processing
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of detection dictionaries with bounding boxes and labels
        """
        # Force GPU usage for inference
        if self.device == 'cuda':
            # Run inference on GPU with optimizations using new PyTorch API
            with torch.amp.autocast('cuda'):  # Use mixed precision for speed
                results = self.model(image, conf=self.confidence_threshold, verbose=False, device='cuda')
        else:
            # CPU inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                # Get image dimensions
                img_height, img_width = image.shape[:2]
                
                for box in boxes:
                    # Get bounding box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.current_classes[class_id]
                    
                    # Create detection for visualization
                    detection = {
                        "box_2d": [
                            int((y1 / img_height) * 1000),  # ymin normalized
                            int((x1 / img_width) * 1000),   # xmin normalized  
                            int((y2 / img_height) * 1000),  # ymax normalized
                            int((x2 / img_width) * 1000)    # xmax normalized
                        ],
                        "label": f"{class_name}",
                        "confidence": confidence,
                        "class_id": class_id,
                        "raw_class": class_name
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def draw_detections_fast(self, image, detections):
        """
        Fast detection drawing optimized for real-time display
        
        Args:
            image: OpenCV image
            detections: List of detection dictionaries
            
        Returns:
            Annotated image
        """
        if not detections:
            return image
        
        img_height, img_width = image.shape[:2]
        annotated = image.copy()
        
        # Define colors for different classes
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
        
        for i, det in enumerate(detections):
            # Denormalize coordinates
            y1 = int((det["box_2d"][0] / 1000) * img_height)
            x1 = int((det["box_2d"][1] / 1000) * img_width)  
            y2 = int((det["box_2d"][2] / 1000) * img_height)
            x2 = int((det["box_2d"][3] / 1000) * img_width)
            
            # Get color for this class
            color = colors[det["class_id"] % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label with confidence
            label = f"{det['raw_class']} {det['confidence']:.2f}"
            
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw text background
            cv2.rectangle(annotated, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       font, font_scale, (255, 255, 255), thickness)
        
        return annotated
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_ui_info(self, image):
        """Draw UI information on the image"""
        info_lines = [
            f"FPS: {self.current_fps:.1f}",
            f"Confidence: {self.confidence_threshold:.2f}",
            f"Classes: {len(self.current_classes)}",
            f"Device: {self.device.upper()}"
        ]
        
        # Add GPU memory info if using CUDA
        if self.device == 'cuda' and torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            info_lines.append(f"GPU Mem: {memory_used:.1f}/{memory_total:.1f}GB")
        
        # Draw semi-transparent background (adjust height for extra line)
        overlay = image.copy()
        bg_height = 120 if self.device == 'cpu' else 145
        cv2.rectangle(overlay, (10, 10), (350, bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)  # Green
        
        for i, line in enumerate(info_lines):
            y_pos = 35 + i * 25
            cv2.putText(image, line, (20, y_pos), font, font_scale, color, 2)
    
    def get_custom_classes_interactive(self):
        """Get custom classes from user input - Open Vocabulary Detection!"""
        # Force input if no classes are set
        while not self.current_classes:
            user_input = input("\n Enter what you want to detect: ").strip()
            
            if user_input:
                # Parse user input
                custom_classes = [cls.strip() for cls in user_input.split(',') if cls.strip()]
                if custom_classes:
                    self.set_classes(custom_classes)
                    print(f" Now detecting {len(custom_classes)} types of objects!")
                    return
                else:
                    print(" Please enter valid object names separated by commas")
            else:
                print("You must specify what to detect! YOLO-World needs your vocabulary.")
        
        # If classes already exist, allow updating
        print(f"\nCurrent detection vocabulary: {', '.join(self.current_classes[:5])}")
        if len(self.current_classes) > 5:
            print(f"    ...and {len(self.current_classes) - 5} more")
        
        user_input = input("\nEnter new objects to detect (or press Enter to keep current): ").strip()
        
        if user_input:
            # Parse user input
            custom_classes = [cls.strip() for cls in user_input.split(',') if cls.strip()]
            if custom_classes:
                self.set_classes(custom_classes)
                print(f"Updated to {len(custom_classes)} custom objects!")
            else:
                print("No valid classes entered, keeping current vocabulary")
        else:
            print("Keeping current detection vocabulary")

    def save_current_frame(self, image, detections):
        """Save current frame with detections"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"realtime_yolo_world_{timestamp}.jpg"
        
        # Use utils function for high-quality visualization
        if detections:
            annotated_image = plot_bounding_boxes_opencv(image, detections, save_path=filename)
            print(f"Frame saved with {len(detections)} detections: {filename}")
        else:
            cv2.imwrite(filename, image)
            print(f"Frame saved (no detections): {filename}")
    
    def run_realtime_detection(self):
        """Main real-time detection loop"""
        print(f" Device: {self.device.upper()}")
        
        # Get vocabulary if not set
        if not self.current_classes:
            print("First, tell YOLO-World what you want to detect!")
            self.get_custom_classes_interactive()
        
        print(f"\nCurrently detecting: {len(self.current_classes)} types of objects")
        print(f"    {', '.join(self.current_classes[:3])}")
        if len(self.current_classes) > 3:
            print(f"    ...and {len(self.current_classes) - 3} more")
        
        print("\nControls:")
        print("  'c' - Change detection vocabulary")  
        print("  's' - Save current frame with detections")
        print("  '+' - Increase confidence threshold (+0.1)")
        print("  '-' - Decrease confidence threshold (-0.1)")
        print("\nStarting webcam...")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set webcam properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Main detection loop
        try:
            while True:
                start_time = time.time()
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Run detection
                detections = self.detect_objects_fast(frame)
                
                # Draw detections
                annotated_frame = self.draw_detections_fast(frame, detections)
                
                # Draw UI info
                self.draw_ui_info(annotated_frame)
                
                # Update FPS
                self.update_fps()
                
                # Display frame
                cv2.imshow('Real-Time YOLO-World Detection', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                

                if key == ord('c'):
                    cv2.destroyAllWindows()
                    self.get_custom_classes_interactive()
                    print("Resuming detection...")
                elif key == ord('s'):
                    self.save_current_frame(frame, detections)
                elif key == ord('+') or key == ord('='):
                    self.confidence_threshold = min(0.95, self.confidence_threshold + 0.1)
                    print(f"Confidence threshold: {self.confidence_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    self.confidence_threshold = max(0.05, self.confidence_threshold - 0.1)
                    print(f"Confidence threshold: {self.confidence_threshold:.2f}")
                
                # Control frame rate
                processing_time = time.time() - start_time
                if processing_time < self.frame_delay:
                    time.sleep(self.frame_delay - processing_time)
                
        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam closed")


def main():
    """Main function"""
    try:
        # Initialize with small model for best real-time performance
        # You can change to 'm' or 'l' if you have a powerful GPU
        detector = RealTimeYOLOWorld(
            model_size='l',           # 's' for speed, 'm' or 'l' for accuracy
            confidence_threshold=0.4, # Higher threshold for cleaner results
            target_fps=30            # Target FPS (actual depends on hardware)
        )
        
        # Run real-time detection
        detector.run_realtime_detection()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTroubleshooting:")
        print("- Ensure webcam is connected and not in use")
        print("- Try lowering confidence threshold with '-' key")
        print("- For better performance, ensure GPU drivers are updated")
        print("- Close other camera applications")


if __name__ == "__main__":
    main()