"""
YOLO-World Object Detection Script

This script uses YOLO-World for open-vocabulary object detection.
Unlike traditional YOLO, it can detect ANY object you describe in text prompts,
not limited to the 80 COCO classes.

Requirements:
- ultralytics (with YOLO-World support)
- OpenCV for webcam capture
- PyTorch with CUDA support for GPU acceleration
"""

import cv2
import os
import torch
from datetime import datetime
from ultralytics import YOLOWorld
import numpy as np
from utils import plot_bounding_boxes_opencv

class YOLOWorldDetector:
    """YOLO-World open-vocabulary object detection"""
    
    def __init__(self, model_size='s', confidence_threshold=0.3):
        """
        Initialize YOLO-World model
        
        Args:
            model_size: 's' (small), 'm' (medium), 'l' (large)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        
        # Check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load YOLO-World model
        model_name = f'yolov8{model_size}-world.pt'
        print(f"Loading {model_name}...")
        self.model = YOLOWorld(model_name)
        
        # Move model to GPU if available
        if self.device == 'cuda':
            self.model.to('cuda')
            print("Model moved to GPU")
        
        # Default classes for table objects (you can customize these)
        self.custom_classes = [
            "laptop", "computer", "keyboard", "mouse", "monitor", "screen",
            "phone", "smartphone", "tablet", "notebook", "pen", "pencil",
            "cup", "mug", "coffee cup", "water bottle", "book", "paper",
            "glasses", "headphones", "charger", "cable", "remote control",
            "calculator", "stapler", "scissors", "tape", "marker",
            "wallet", "keys", "watch", "coin", "card", "tissue box",
            "lamp", "plant", "clock", "frame", "speaker", "microphone"
        ]
        
        # Set the custom vocabulary
        self.set_classes(self.custom_classes)
    
    def set_classes(self, class_list):
        """
        Set custom classes for detection
        
        Args:
            class_list: List of class names to detect
        """
        print(f"Setting custom vocabulary with {len(class_list)} classes:")
        print(f"Classes: {', '.join(class_list[:10])}...")  # Show first 10
        
        self.model.set_classes(class_list)
        self.current_classes = class_list
    
    def capture_webcam(self):
        """Capture image from webcam (same as traditional YOLO)"""
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
        Run YOLO-World detection on image
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of detection dictionaries with bounding boxes and labels
        """
        print("Running YOLO-World detection...")
        print(f"Detecting from vocabulary: {len(self.current_classes)} classes")
        
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
                    class_name = self.current_classes[class_id]
                    
                    # Create detection in same format as Gemini
                    detection = {
                        "box_2d": [y1_norm, x1_norm, y2_norm, x2_norm],  # [ymin, xmin, ymax, xmax]
                        "label": f"{class_name}_{confidence:.2f}",
                        "confidence": confidence,
                        "class_id": class_id,
                        "raw_class": class_name
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
        output_filename = f"yolo_world_detections_{timestamp}.jpg"
        
        # Use utils function for visualization
        annotated_image = plot_bounding_boxes_opencv(image, detections, save_path=output_filename)
        
        print(f"💾 Annotated image saved as: {output_filename}")
        
        # Display the image
        cv2.imshow('YOLO-World Detection Results - Press any key to close', annotated_image)
        print("\nPress any key in the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def capture_and_detect(self, custom_classes=None):
        """Complete workflow: capture image and run YOLO-World detection"""
        print("YOLO-World Open-Vocabulary Object Detection")
        print("=" * 45)
        print(f"Device: {self.device}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        
        # Allow custom class input
        if custom_classes:
            self.set_classes(custom_classes)
        
        print()
        
        # Step 1: Capture image
        print("Step 1: Capturing image from webcam...")
        captured_frame = self.capture_webcam()
        
        if captured_frame is None:
            print("No image captured. Exiting.")
            return
        
        print()
        
        # Step 2: Run YOLO-World detection
        print("Step 2: Running YOLO-World detection...")
        detections = self.detect_objects(captured_frame)
        
        # Step 3: Display results
        print("\n" + "="*50)
        print("DETECTION RESULTS")
        print("="*50)
        
        if detections:
            print(f"Detected {len(detections)} objects:")
            for i, det in enumerate(detections, 1):
                print(f"{i}. {det['raw_class']} (confidence: {det['confidence']:.2f})")
            
            print("\nStep 3: Displaying detections on image...")
            self.display_results(captured_frame, detections)
        else:
            print("No objects detected above confidence threshold")
            print("Try:")
            print("- Lowering confidence threshold")
            print("- Adding more specific class names")
            print("- Ensuring objects are clearly visible")
        
        print("\n" + "="*50)


def get_custom_classes():
    """Interactive function to get custom classes from user"""
    print("\nCustom Class Input:")
    print("Enter objects you want to detect (comma-separated)")
    print("Example: laptop, mouse, coffee cup, smartphone, pen")
    print("Or press Enter to use default table objects")
    
    user_input = input("\nEnter classes: ").strip()
    
    if user_input:
        # Parse user input
        custom_classes = [cls.strip() for cls in user_input.split(',') if cls.strip()]
        print(f"Using custom classes: {custom_classes}")
        return custom_classes
    else:
        print("Using default table object classes")
        return None


def main():
    """Main function"""
    try:
        # Initialize YOLO-World
        print("Initializing YOLO-World...")
        
        # You can change model size: 's', 'm', 'l'
        yolo_world = YOLOWorldDetector(model_size='s', confidence_threshold=0.3)
        
        # Get custom classes from user
        custom_classes = get_custom_classes()
        
        # Run detection workflow
        yolo_world.capture_and_detect(custom_classes)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTroubleshooting:")
        print("- Make sure you have ultralytics with YOLO-World: pip install ultralytics")
        print("- For GPU acceleration, install PyTorch with CUDA support")
        print("- Make sure your webcam is connected and not in use by another app")
        print("- Try lowering confidence threshold if no detections")


if __name__ == "__main__":
    main()