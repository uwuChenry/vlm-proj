"""
Traditional YOLO Object Detection Script

This script uses YOLOv8 (ultralytics) to run traditional object detection locally on GPU.
It detects the standard 80 COCO classes that YOLO was trained on.

Requirements:
- ultralytics (YOLOv8)
- OpenCV for webcam capture
- PyTorch with CUDA support for GPU acceleration
"""

import cv2
import os
import torch
from datetime import datetime
from ultralytics import YOLO
import numpy as np
from utils import plot_bounding_boxes_opencv

class TraditionalYOLO:
    """Traditional YOLO object detection using YOLOv8"""
    
    def __init__(self, model_size='n', confidence_threshold=0.5):
        """
        Initialize YOLO model
        
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        
        # Check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load YOLOv8 model
        model_name = f'yolov8{model_size}.pt'
        print(f"Loading {model_name}...")
        self.model = YOLO(model_name)
        
        # Move model to GPU if available
        if self.device == 'cuda':
            self.model.to('cuda')
            print("Model moved to GPU")
        
        # COCO class names (80 classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def capture_webcam(self):
        """Capture image from webcam (same as before)"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return None
        
        print("Webcam opened successfully!")
        print("Controls:")
        print("- Press 'c' to capture image")
        print("- Press 'q' to quit")
        print()
        
        captured_frame = None
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            cv2.imshow('Webcam - Press "c" to capture, "q" to quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captured_image_{timestamp}.jpg"
                
                cv2.imwrite(filename, frame)
                print(f"📸 Image saved as: {filename}")
                
                captured_frame = frame.copy()
                print("Image captured! Closing camera...")
                break
                
            elif key == ord('q'):
                print("Exiting...")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")
        
        return captured_frame
    
    def detect_objects(self, image):
        """
        Run YOLO detection on image
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of detection dictionaries with bounding boxes and labels
        """
        print("Running YOLO detection...")
        
        # Run inference
        results = self.model(image, conf=self.confidence_threshold)
        
        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                # Get image dimensions for normalization
                img_height, img_width = image.shape[:2]
                
                for box in boxes:
                    # Get bounding box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Normalize coordinates to 0-1000 range
                    x1_norm = int((x1 / img_width) * 1000)
                    y1_norm = int((y1 / img_height) * 1000)
                    x2_norm = int((x2 / img_width) * 1000)
                    y2_norm = int((y2 / img_height) * 1000)
                    
                    # Get confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.class_names[class_id]
                    
                    # Create detection in same format as Gemini
                    detection = {
                        "box_2d": [y1_norm, x1_norm, y2_norm, x2_norm],  # [ymin, xmin, ymax, xmax]
                        "label": f"{class_name}_{confidence:.2f}",
                        "confidence": confidence,
                        "class_id": class_id
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def display_results(self, image, detections):
        """Display detection results on image"""
        if not detections:
            print("No objects detected")
            return
        
        print(f"Found {len(detections)} detections")
        
        # Generate save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"yolo_traditional_detections_{timestamp}.jpg"
        
        # Use utils function for visualization
        annotated_image = plot_bounding_boxes_opencv(image, detections, save_path=output_filename)
        
        print(f"💾 Annotated image saved as: {output_filename}")
        
        # Display the image
        cv2.imshow('YOLO Traditional Detection Results - Press any key to close', annotated_image)
        print("\nPress any key in the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def capture_and_detect(self):
        """Complete workflow: capture image and run YOLO detection"""
        print("Traditional YOLO Object Detection")
        print("=" * 35)
        print(f"Device: {self.device}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print()
        
        # Step 1: Capture image
        print("Step 1: Capturing image from webcam...")
        captured_frame = self.capture_webcam()
        
        if captured_frame is None:
            print("No image captured. Exiting.")
            return
        
        print()
        
        # Step 2: Run YOLO detection
        print("Step 2: Running YOLO detection...")
        detections = self.detect_objects(captured_frame)
        
        # Step 3: Display results
        print("\n" + "="*50)
        print("DETECTION RESULTS")
        print("="*50)
        
        if detections:
            print(f"Detected {len(detections)} objects:")
            for i, det in enumerate(detections, 1):
                print(f"{i}. {det['label']} (confidence: {det['confidence']:.2f})")
            
            print("\nStep 3: Displaying detections on image...")
            self.display_results(captured_frame, detections)
        else:
            print("No objects detected above confidence threshold")
        
        print("\n" + "="*50)


def main():
    """Main function"""
    try:
        # Initialize traditional YOLO
        print("Initializing Traditional YOLO (YOLOv8)...")
        
        # You can change model size: 'n', 's', 'm', 'l', 'x' (nano to extra-large)
        # Larger models are more accurate but slower
        yolo = TraditionalYOLO(model_size='n', confidence_threshold=0.5)
        
        # Run detection workflow
        yolo.capture_and_detect()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTroubleshooting:")
        print("- Make sure you have ultralytics installed: pip install ultralytics")
        print("- For GPU acceleration, install PyTorch with CUDA support")
        print("- Make sure your webcam is connected and not in use by another app")


if __name__ == "__main__":
    main()