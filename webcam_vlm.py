"""
Webcam VLM Scene Description Script

This script captures an image from a USB webcam and sends it to Google Gemini
for scene description and analysis.

Requirements:
- OpenCV for webcam capture
- Google API key for Gemini
- PIL/Pillow for image processing
"""

import cv2
import os
import json
from datetime import datetime
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from utils import plot_points_opencv, plot_bounding_boxes_opencv, parse_json, display_image_with_detections

# Load environment variables from .env file
load_dotenv()

def capture_webcam():
    """Webcam capture function (copied from working simple_webcam.py)"""
    
    # Initialize the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("Make sure your webcam is connected and not being used by another app")
        return None
    
    print("Webcam opened successfully!")
    print("Controls:")
    print("- Press 'c' to capture image")
    print("- Press 'q' to quit")
    print()
    
    captured_frame = None
    
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
            filename = f"captured_image_{timestamp}.jpg"
            
            # Save the image
            cv2.imwrite(filename, frame)
            print(f"📸 Image saved as: {filename}")
            
            # Store the captured frame
            captured_frame = frame.copy()
            print("Image captured! Closing camera...")
            break
            
        elif key == ord('q'):  # Quit
            print("Exiting...")
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")
    
    return captured_frame
#old prompt: Detect and identify objects in this image. The label returned should be an identifying name for the object detected. The answer should follow the json format: [{\"point\": <point>, \"label\": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000.
def analyze_with_gemini(image_frame, prompt="Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects. Include as many objects as you can identify on the table. If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..). The format should be as follows: [{\"box_2d\": [ymin, xmin, ymax, xmax], \"label\": <label for the object>}] normalized to 0-1000. The values in box_2d must only be integers"):
    """Send image to Google Gemini for analysis"""
    try:
        # Get API key from environment
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return {
                "success": False,
                "error": "Google API key not found. Make sure GOOGLE_API_KEY is set in your .env file"
            }
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-robotics-er-1.5-preview')
        
        # Save the OpenCV image to a temporary file first
        temp_filename = "temp_for_gemini.jpg"
        cv2.imwrite(temp_filename, image_frame)
        
        # Load the image using PIL
        pil_image = Image.open(temp_filename)
        
        print("Sending image to Google Gemini for analysis...")
        
        # Generate content using Gemini
        response = model.generate_content([prompt, pil_image])
        
        # Clean up temporary file
        try:
            os.remove(temp_filename)
        except:
            pass  # Don't worry if we can't delete the temp file
        
        return {
            "success": True,
            "analysis": response.text,
            "provider": "gemini",
            "model_used": "gemini-2.5-flash"
        }
        
    except Exception as e:
        # Clean up temporary file in case of error
        try:
            os.remove("temp_for_gemini.jpg")
        except:
            pass
            
            return {
                "success": False,
                "error": f"Error analyzing with Gemini: {str(e)}"
            }

def parse_detection_results(analysis_text):
    """Parse the JSON response from Gemini and extract object detections"""
    try:
        # Use the improved parse_json from utils
        json_str = parse_json(analysis_text)
        detections = json.loads(json_str)
        return detections
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {analysis_text}")
        return []

def display_results_on_image(image, result):
    """Display the analysis results on the image and show it"""
    if not result["success"]:
        print("Cannot display results - analysis failed")
        return
    
    # Parse the detection results
    detections = parse_detection_results(result["analysis"])
    
    if detections:
        print(f"Found {len(detections)} detections")
        
        # Determine detection type based on data structure
        detection_type = "points"
        if detections and isinstance(detections[0], dict):
            if 'box_2d' in detections[0]:
                detection_type = "bounding_boxes"
            elif 'point' in detections[0]:
                detection_type = "points"
        
        print(f"Using detection type: {detection_type}")
        
        # Generate save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"annotated_image_{detection_type}_{timestamp}.jpg"
        
        # Use the improved utils functions for visualization
        annotated_image = display_image_with_detections(
            image, detections, detection_type, save_path=output_filename
        )
        
        print(f"💾 Annotated image saved as: {output_filename}")
        
        # Display the image
        cv2.imshow('Object Detection Results - Press any key to close', annotated_image)
        print("\nPress any key in the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No valid detections found in response")

def main():
    """Main function"""
    print("Webcam VLM Scene Description Tool")
    print("=================================")
    print("Using Google Gemini (FREE)")
    print()
    
    try:
        # Step 1: Capture image from webcam
        print("Step 1: Capturing image from webcam...")
        captured_frame = capture_webcam()
        
        if captured_frame is None:
            print("No image captured. Exiting.")
            return
        
        print()
        
        # Step 2: Analyze with Gemini using object detection prompt
        print("Step 2: Analyzing with Gemini...")
        result = analyze_with_gemini(captured_frame)
        
        # Step 3: Display results
        print("\n" + "="*50)
        print("ANALYSIS RESULTS")
        print("="*50)
        
        if result["success"]:
            print(f"Provider: {result['provider'].upper()}")
            print(f"Model used: {result['model_used']}")
            print("\nDetection Results:")
            print("-" * 20)
            print(result["analysis"])
            
            # Step 4: Display results on image
            print("\nStep 3: Displaying detections on image...")
            display_results_on_image(captured_frame, result)
            
        else:
            print(f"Error: {result['error']}")
            print("\nTip: Make sure you have GOOGLE_API_KEY set in your .env file")
            print("Get a free key at: https://makersuite.google.com/app/apikey")
        
        print("\n" + "="*50)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()