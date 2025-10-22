"""
Simple Webcam Capture Script

A basic script that opens your webcam and takes pictures when you press 'c'.
Press 'q' to quit.
"""

import cv2
import os
from datetime import datetime

def capture_webcam():
    """Simple webcam capture function"""
    
    # Initialize the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("Make sure your webcam is connected and not being used by another app")
        return
    
    print("Webcam opened successfully!")
    print("Controls:")
    print("- Press 'c' to capture image")
    print("- Press 'q' to quit")
    print()
    
    # Counter for image numbering
    image_count = 1
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        # Display the frame
        cv2.imshow('Webcam - Press "c" to capture, "q" to quit', frame)
        
        # Wait for key press (1ms timeout)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):  # Capture image
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_image_{image_count}_{timestamp}.jpg"
            
            # Save the image
            cv2.imwrite(filename, frame)
            print(f"📸 Image saved as: {filename}")
            
            # Increment counter
            image_count += 1
            
        elif key == ord('q'):  # Quit
            print("Exiting...")
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

def main():
    """Main function"""
    print("Simple Webcam Capture Tool")
    print("=" * 30)
    
    try:
        capture_webcam()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()