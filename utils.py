"""
Utility functions for object detection visualization
Adapted from Google's vision utilities for webcam VLM project
"""

import base64
import json
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
from io import BytesIO
from typing import List, Dict, Any, Tuple
import cv2


def parse_json(response_text):
    """Parse JSON from response text, handling markdown fencing"""
    try:
        # Remove markdown code fencing if present
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.rfind("```")
            json_str = response_text[start:end].strip()
        else:
            # Try to find JSON array in the text
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
            else:
                json_str = response_text
        
        return json_str
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return response_text


def plot_points_opencv(img, points_data, save_path=None):
    """
    Plot points on OpenCV image (BGR format)
    
    Args:
        img: OpenCV image (BGR format)
        points_data: List of detection dictionaries with 'point' and 'label'
        save_path: Optional path to save the annotated image
    
    Returns:
        Annotated OpenCV image
    """
    if isinstance(points_data, str):
        points_data = json.loads(parse_json(points_data))
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Create a copy to draw on
    annotated_img = img.copy()
    
    # Define colors (BGR format for OpenCV)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 255),  # Orange-red
        (128, 128, 128) # Gray
    ]
    
    for i, detection in enumerate(points_data):
        try:
            point = detection.get('point', [])
            label = detection.get('label', f'Object_{i}')
            
            if len(point) >= 2:
                # Convert normalized coordinates (0-1000) to pixel coordinates
                y_norm, x_norm = point[0], point[1]
                x_pixel = int((x_norm / 1000.0) * width)
                y_pixel = int((y_norm / 1000.0) * height)
                
                # Select color
                color = colors[i % len(colors)]
                
                # Draw circle at detection point
                cv2.circle(annotated_img, (x_pixel, y_pixel), 8, color, -1)
                cv2.circle(annotated_img, (x_pixel, y_pixel), 10, color, 2)
                
                # Draw label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # Draw background rectangle
                cv2.rectangle(annotated_img,
                            (x_pixel - 5, y_pixel - text_height - 15),
                            (x_pixel + text_width + 5, y_pixel - 5),
                            color, -1)
                
                # Draw text
                cv2.putText(annotated_img, label,
                          (x_pixel, y_pixel - 10),
                          font, font_scale, (255, 255, 255), thickness)
                
                print(f"Drew point: {label} at ({x_pixel}, {y_pixel})")
                
        except Exception as e:
            print(f"Error drawing detection {i}: {e}")
            continue
    
    if save_path:
        cv2.imwrite(save_path, annotated_img)
        print(f"Saved annotated image to: {save_path}")
    
    return annotated_img


def plot_bounding_boxes_opencv(img, bounding_boxes_data, save_path=None):
    """
    Plot bounding boxes on OpenCV image (BGR format)
    
    Args:
        img: OpenCV image (BGR format)
        bounding_boxes_data: List of detection dictionaries with 'box_2d' and 'label'
        save_path: Optional path to save the annotated image
    
    Returns:
        Annotated OpenCV image
    """
    if isinstance(bounding_boxes_data, str):
        bounding_boxes_data = json.loads(parse_json(bounding_boxes_data))
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Create a copy to draw on
    annotated_img = img.copy()
    
    # Define colors (BGR format for OpenCV)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 255),  # Orange-red
        (128, 128, 128) # Gray
    ]
    
    for i, detection in enumerate(bounding_boxes_data):
        try:
            box_2d = detection.get('box_2d', [])
            label = detection.get('label', f'Object_{i}')
            
            if len(box_2d) >= 4:
                # Convert normalized coordinates (0-1000) to pixel coordinates
                # box_2d format: [y1, x1, y2, x2]
                y1_norm, x1_norm, y2_norm, x2_norm = box_2d[:4]
                
                x1_pixel = int((x1_norm / 1000.0) * width)
                y1_pixel = int((y1_norm / 1000.0) * height)
                x2_pixel = int((x2_norm / 1000.0) * width)
                y2_pixel = int((y2_norm / 1000.0) * height)
                
                # Ensure coordinates are in correct order
                x1_pixel, x2_pixel = min(x1_pixel, x2_pixel), max(x1_pixel, x2_pixel)
                y1_pixel, y2_pixel = min(y1_pixel, y2_pixel), max(y1_pixel, y2_pixel)
                
                # Select color
                color = colors[i % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(annotated_img,
                            (x1_pixel, y1_pixel),
                            (x2_pixel, y2_pixel),
                            color, 3)
                
                # Draw label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # Draw background rectangle for text
                cv2.rectangle(annotated_img,
                            (x1_pixel, y1_pixel - text_height - 10),
                            (x1_pixel + text_width + 10, y1_pixel),
                            color, -1)
                
                # Draw text
                cv2.putText(annotated_img, label,
                          (x1_pixel + 5, y1_pixel - 5),
                          font, font_scale, (255, 255, 255), thickness)
                
                print(f"Drew bounding box: {label} at ({x1_pixel},{y1_pixel}) to ({x2_pixel},{y2_pixel})")
                
        except Exception as e:
            print(f"Error drawing bounding box {i}: {e}")
            continue
    
    if save_path:
        cv2.imwrite(save_path, annotated_img)
        print(f"Saved annotated image to: {save_path}")
    
    return annotated_img


def plot_points_pil(pil_image, points_data, save_path=None):
    """
    Plot points on PIL image
    
    Args:
        pil_image: PIL Image object
        points_data: List of detection dictionaries with 'point' and 'label'
        save_path: Optional path to save the annotated image
    
    Returns:
        Annotated PIL Image
    """
    if isinstance(points_data, str):
        points_data = json.loads(parse_json(points_data))
    
    # Get image dimensions
    width, height = pil_image.size
    
    # Create a copy to draw on
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    
    # Define colors
    colors = [
        "green", "blue", "red", "yellow", "magenta", "cyan", 
        "purple", "orange", "pink", "brown", "gray", "lime"
    ]
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for i, detection in enumerate(points_data):
        try:
            point = detection.get('point', [])
            label = detection.get('label', f'Object_{i}')
            
            if len(point) >= 2:
                # Convert normalized coordinates (0-1000) to pixel coordinates
                y_norm, x_norm = point[0], point[1]
                x_pixel = int((x_norm / 1000.0) * width)
                y_pixel = int((y_norm / 1000.0) * height)
                
                # Select color
                color = colors[i % len(colors)]
                
                # Draw circle at detection point
                radius = 8
                draw.ellipse([x_pixel - radius, y_pixel - radius,
                            x_pixel + radius, y_pixel + radius],
                           fill=color, outline=color)
                
                # Draw label with background
                bbox = draw.textbbox((x_pixel + 12, y_pixel - 8), label, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x_pixel + 12, y_pixel - 8), label, fill="white", font=font)
                
                print(f"Drew point: {label} at ({x_pixel}, {y_pixel})")
                
        except Exception as e:
            print(f"Error drawing detection {i}: {e}")
            continue
    
    if save_path:
        img.save(save_path)
        print(f"Saved annotated image to: {save_path}")
    
    return img


def plot_bounding_boxes_pil(pil_image, bounding_boxes_data, save_path=None):
    """
    Plot bounding boxes on PIL image
    
    Args:
        pil_image: PIL Image object
        bounding_boxes_data: List of detection dictionaries with 'box_2d' and 'label'
        save_path: Optional path to save the annotated image
    
    Returns:
        Annotated PIL Image
    """
    if isinstance(bounding_boxes_data, str):
        bounding_boxes_data = json.loads(parse_json(bounding_boxes_data))
    
    # Get image dimensions
    width, height = pil_image.size
    
    # Create a copy to draw on
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    
    # Define colors
    colors = [
        "green", "blue", "red", "yellow", "magenta", "cyan", 
        "purple", "orange", "pink", "brown", "gray", "lime"
    ]
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for i, detection in enumerate(bounding_boxes_data):
        try:
            box_2d = detection.get('box_2d', [])
            label = detection.get('label', f'Object_{i}')
            
            if len(box_2d) >= 4:
                # Convert normalized coordinates (0-1000) to pixel coordinates
                # box_2d format: [y1, x1, y2, x2]
                y1_norm, x1_norm, y2_norm, x2_norm = box_2d[:4]
                
                x1_pixel = int((x1_norm / 1000.0) * width)
                y1_pixel = int((y1_norm / 1000.0) * height)
                x2_pixel = int((x2_norm / 1000.0) * width)
                y2_pixel = int((y2_norm / 1000.0) * height)
                
                # Ensure coordinates are in correct order
                x1_pixel, x2_pixel = min(x1_pixel, x2_pixel), max(x1_pixel, x2_pixel)
                y1_pixel, y2_pixel = min(y1_pixel, y2_pixel), max(y1_pixel, y2_pixel)
                
                # Select color
                color = colors[i % len(colors)]
                
                # Draw bounding box
                draw.rectangle([x1_pixel, y1_pixel, x2_pixel, y2_pixel],
                             outline=color, width=3)
                
                # Draw label with background
                bbox = draw.textbbox((x1_pixel + 5, y1_pixel - 25), label, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1_pixel + 5, y1_pixel - 25), label, fill="white", font=font)
                
                print(f"Drew bounding box: {label} at ({x1_pixel},{y1_pixel}) to ({x2_pixel},{y2_pixel})")
                
        except Exception as e:
            print(f"Error drawing bounding box {i}: {e}")
            continue
    
    if save_path:
        img.save(save_path)
        print(f"Saved annotated image to: {save_path}")
    
    return img


def convert_opencv_to_pil(opencv_img):
    """Convert OpenCV image (BGR) to PIL image (RGB)"""
    rgb_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)


def convert_pil_to_opencv(pil_img):
    """Convert PIL image (RGB) to OpenCV image (BGR)"""
    rgb_array = np.array(pil_img)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def display_image_with_detections(img, detections, detection_type="points", save_path=None):
    """
    Unified function to display image with detections
    
    Args:
        img: OpenCV image (BGR) or PIL image
        detections: Detection data (string or list)
        detection_type: "points" or "bounding_boxes"
        save_path: Optional path to save annotated image
    
    Returns:
        Annotated image (same format as input)
    """
    is_opencv = isinstance(img, np.ndarray)
    
    if detection_type == "points":
        if is_opencv:
            return plot_points_opencv(img, detections, save_path)
        else:
            return plot_points_pil(img, detections, save_path)
    elif detection_type == "bounding_boxes":
        if is_opencv:
            return plot_bounding_boxes_opencv(img, detections, save_path)
        else:
            return plot_bounding_boxes_pil(img, detections, save_path)
    else:
        raise ValueError("detection_type must be 'points' or 'bounding_boxes'")