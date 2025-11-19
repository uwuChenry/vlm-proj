"""
Webcam VLM Segmentation Testing Script

This script captures an image from a USB webcam and tests if Google Gemini
can return segmentation masks along with object detection.

Requirements:
- OpenCV for webcam capture
- Google API key for Gemini
- PIL/Pillow for image processing
- Base64 for mask decoding
"""

import cv2
import os
import json
import base64
import numpy as np
from datetime import datetime
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from utils import plot_points_opencv, plot_bounding_boxes_opencv, parse_json, display_image_with_detections, plot_segmentation_opencv

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
            print(f"Image saved as: {filename}")
            
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

def analyze_with_gemini_segmentation(image_frame):
    """Send image to Google Gemini for segmentation analysis"""
    
    # Segmentation prompt - asking for masks in addition to bounding boxes
    segmentation_prompt = """
    Detect and segment objects in this image. Return both bounding boxes and segmentation masks as a JSON array.
    
    For each object, provide:
    - Bounding box coordinates
    - Object label
    - Segmentation mask as a base64-encoded PNG image
    
    The format should be:
    [
        {
            "box_2d": [ymin, xmin, ymax, xmax],
            "label": "object_name",
            "mask": "data:image/png;base64,<base64_encoded_mask>"
        }
    ]
    
    Coordinates should be normalized to 0-1000 as integers.
    Masks should be binary (black/white) PNG images matching the original image dimensions.
    Limit to 15 most prominent objects.
    Never return code fencing or explanatory text, only valid JSON.
    """
    
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
        temp_filename = "temp_for_gemini_segmentation.jpg"
        cv2.imwrite(temp_filename, image_frame)
        
        # Load the image using PIL
        pil_image = Image.open(temp_filename)
        
        print("Sending image to Google Gemini for segmentation analysis...")
        print("Testing if Gemini can return segmentation masks...")
        
        # Generate content using Gemini
        response = model.generate_content([segmentation_prompt, pil_image])
        print("Received response from Gemini.")
        # Clean up temporary file
        try:
            os.remove(temp_filename)
        except:
            pass  # Don't worry if we can't delete the temp file
        
        return {
            "success": True,
            "analysis": response.text,
            "provider": "gemini",
            "model_used": "gemini-robotics-er-1.5-preview",
            "type": "segmentation"
        }
        
    except Exception as e:
        # Clean up temporary file in case of error
        try:
            os.remove("temp_for_gemini_segmentation.jpg")
        except:
            pass
            
        return {
            "success": False,
            "error": f"Error analyzing with Gemini: {str(e)}"
        }

def parse_segmentation_results(analysis_text):
    """Parse the JSON response from Gemini and extract segmentation data"""
    try:
        # Use the improved parse_json from utils
        json_str = parse_json(analysis_text)
        segmentations = json.loads(json_str)
        
        # Validate segmentation format
        valid_segmentations = []
        for seg in segmentations:
            if isinstance(seg, dict) and 'box_2d' in seg and 'label' in seg:
                # Check if mask is present
                if 'mask' in seg:
                    print(f"Found segmentation with mask for: {seg['label']}")
                    valid_segmentations.append(seg)
                else:
                    print(f"Found detection without mask for: {seg['label']}")
                    # Still add it as a regular detection
                    valid_segmentations.append(seg)
        
        return valid_segmentations
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {analysis_text}")
        return []

def display_segmentation_results(image, result):
    """Display the segmentation results on the image"""
    if not result["success"]:
        print("Cannot display results - analysis failed")
        return
    
    # Parse the segmentation results
    segmentations = parse_segmentation_results(result["analysis"])
    
    if segmentations:
        print(f"Found {len(segmentations)} segmentations/detections")
        
        # Check if any have masks
        has_masks = any('mask' in seg for seg in segmentations)
        
        if has_masks:
            print("SUCCESS: Gemini returned segmentation masks!")
            detection_type = "segmentation"
        else:
            print("INFO: Gemini returned detections but no masks")
            detection_type = "bounding_boxes"
        
        # Generate save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"segmentation_test_{timestamp}.jpg"
        
        # Use appropriate visualization function
        if detection_type == "segmentation":
            annotated_image = plot_segmentation_opencv(image, segmentations, save_path=output_filename)
        else:
            annotated_image = display_image_with_detections(
                image, segmentations, "bounding_boxes", save_path=output_filename
            )
        
        print(f"Annotated image saved as: {output_filename}")
        
        # Display the image
        window_title = 'Segmentation Test Results - Press any key to close'
        cv2.imshow(window_title, annotated_image)
        print("\nPress any key in the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return has_masks
        
    else:
        print("No valid segmentations found in response")
        return False

def main():
    """Main function"""
    print("Webcam VLM Segmentation Testing Tool")
    print("====================================")
    print("Testing if Google Gemini can return segmentation masks")
    print()
    
    try:
        # Step 1: Capture image from webcam
        print("Step 1: Capturing image from webcam...")
        captured_frame = capture_webcam()
        
        if captured_frame is None:
            print("No image captured. Exiting.")
            return
        
        print()
        
        # Step 2: Analyze with Gemini using segmentation prompt
        print("Step 2: Testing Gemini segmentation capabilities...")
        result = analyze_with_gemini_segmentation(captured_frame)
        
        # Step 3: Display results
        print("\n" + "="*60)
        print("SEGMENTATION TEST RESULTS")
        print("="*60)
        
        if result["success"]:
            print(f"Provider: {result['provider'].upper()}")
            print(f"Model used: {result['model_used']}")
            print(f"Test type: {result['type']}")
            print("\nRaw Response:")
            print("-" * 20)
            print(result["analysis"])
            print("-" * 20)
            
            # Step 4: Display results on image
            print("\nStep 3: Processing and displaying results...")
            has_segmentation = display_segmentation_results(captured_frame, result)
            
            # Summary
            print("\n" + "="*60)
            print("SEGMENTATION CAPABILITY SUMMARY")
            print("="*60)
            
            if has_segmentation:
                print("✅ SUCCESS: Gemini CAN return segmentation masks!")
                print("   - Returned pixel-level segmentation data")
                print("   - Masks can be decoded and visualized")
                print("   - Ready for advanced computer vision tasks")
            else:
                print("❌ LIMITATION: Gemini returned detections but NO masks")
                print("   - Only bounding boxes available")
                print("   - No pixel-level segmentation data")
                print("   - Consider using specialized segmentation models")
            
        else:
            print(f"Error: {result['error']}")
            print("\nTip: Make sure you have GOOGLE_API_KEY set in your .env file")
            print("Get a free key at: https://makersuite.google.com/app/apikey")
        
        print("\n" + "="*60)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()