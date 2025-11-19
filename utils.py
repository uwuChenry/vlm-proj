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
from datetime import datetime


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
                
                # Draw label with smaller font
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4  # Changed from 0.6 to 0.4 for smaller text
                thickness = 1     # Changed from 2 to 1 for thinner text
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # Draw background rectangle
                cv2.rectangle(annotated_img,
                            (x_pixel - 5, y_pixel - text_height - 15),
                            (x_pixel + text_width + 5, y_pixel - 5),
                            color, -1)
                
                # Draw text with smaller, thinner style
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
                
                # Draw bounding box with thinner lines
                cv2.rectangle(annotated_img,
                            (x1_pixel, y1_pixel),
                            (x2_pixel, y2_pixel),
                            color, 2)  # Changed from 3 to 2 for thinner lines
                
                # Draw label with smaller font
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5  # Changed from 0.7 to 0.5 for smaller text
                thickness = 1     # Changed from 2 to 1 for thinner text
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # Draw background rectangle for text
                cv2.rectangle(annotated_img,
                            (x1_pixel, y1_pixel - text_height - 10),
                            (x1_pixel + text_width + 10, y1_pixel),
                            color, -1)
                
                # Draw text with smaller, thinner style  
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
    
    # Try to load Calibri font, fallback to other fonts if not available
    try:
        font = ImageFont.truetype("calibri.ttf", 12)  # Smaller Calibri font
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 12)  # Fallback to Arial
        except:
            font = ImageFont.load_default()  # Last resort default font
    
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
    
    # Try to load Calibri font, fallback to other fonts if not available
    try:
        font = ImageFont.truetype("calibri.ttf", 12)  # Smaller Calibri font
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 12)  # Fallback to Arial
        except:
            font = ImageFont.load_default()  # Last resort default font
    
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
                
                # Draw bounding box with thinner lines
                draw.rectangle([x1_pixel, y1_pixel, x2_pixel, y2_pixel],
                             outline=color, width=2)  # Changed from 3 to 2 for thinner lines
                
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
    elif detection_type == "segmentation":
        if is_opencv:
            return plot_segmentation_opencv(img, detections, save_path)
        else:
            return plot_segmentation_pil(img, detections, save_path)
    else:
        raise ValueError("detection_type must be 'points', 'bounding_boxes', or 'segmentation'")


def decode_segmentation_mask(mask_data):
    """
    Decode base64-encoded PNG mask data to numpy array
    
    Args:
        mask_data: Base64-encoded PNG mask string (with or without data URI prefix)
    
    Returns:
        numpy array representing the binary mask
    """
    try:
        # Debug: Print first 100 characters of mask data
        print(f"Mask data preview: {str(mask_data)[:100]}...")
        print(f"Mask data type: {type(mask_data)}")
        print(f"Mask data length: {len(str(mask_data))}")
        
        # Ensure we have a string
        if not isinstance(mask_data, str):
            print(f"Converting mask_data from {type(mask_data)} to string")
            mask_data = str(mask_data)
        
        # Remove data URI prefix if present
        if mask_data.startswith('data:image/png;base64,'):
            mask_data = mask_data.replace('data:image/png;base64,', '')
            print("Removed data URI prefix")
        
        # Clean up the base64 string (remove whitespace, newlines)
        mask_data = mask_data.strip().replace('\n', '').replace('\r', '').replace(' ', '')
        print(f"Cleaned mask data length: {len(mask_data)}")
        
        # Validate base64 format
        if len(mask_data) % 4 != 0:
            # Add padding if needed
            padding = 4 - (len(mask_data) % 4)
            mask_data += '=' * padding
            print(f"Added {padding} padding characters")
        
        # Decode base64
        try:
            mask_bytes = base64.b64decode(mask_data)
            print(f"Successfully decoded base64, got {len(mask_bytes)} bytes")
            
            # Save raw bytes to file for debugging
            debug_filename = f"debug_mask_{datetime.now().strftime('%H%M%S')}.bin"
            with open(debug_filename, 'wb') as f:
                f.write(mask_bytes)
            print(f"Saved raw mask bytes to {debug_filename}")
            
        except Exception as decode_error:
            print(f"Base64 decode error: {decode_error}")
            return None
        
        # Check if we have valid image data
        if len(mask_bytes) < 8:
            print(f"Invalid image data: only {len(mask_bytes)} bytes")
            return None
        
        # Check PNG signature
        png_signature = b'\x89PNG\r\n\x1a\n'
        if not mask_bytes.startswith(png_signature):
            print(f"Not a valid PNG file. First 16 bytes: {mask_bytes[:16]}")
            print(f"First 16 bytes as hex: {mask_bytes[:16].hex()}")
            
            # Check for other image formats
            if mask_bytes.startswith(b'\xff\xd8\xff'):
                print("Detected JPEG format")
                try:
                    mask_image = Image.open(BytesIO(mask_bytes))
                    mask_array = np.array(mask_image.convert('L'))
                    binary_mask = (mask_array > 128).astype(np.uint8)
                    return binary_mask
                except Exception as jpeg_error:
                    print(f"JPEG processing failed: {jpeg_error}")
            
            elif mask_bytes.startswith(b'GIF8'):
                print("Detected GIF format")
                try:
                    mask_image = Image.open(BytesIO(mask_bytes))
                    mask_array = np.array(mask_image.convert('L'))
                    binary_mask = (mask_array > 128).astype(np.uint8)
                    return binary_mask
                except Exception as gif_error:
                    print(f"GIF processing failed: {gif_error}")
            
            # Try to interpret as raw binary data
            print("Attempting to interpret as raw binary mask data...")
            try:
                # Assume it's raw binary data, try different interpretations
                mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
                print(f"Raw binary data length: {len(mask_array)}")
                
                # Try to find a reasonable image size
                total_pixels = len(mask_array)
                possible_sizes = []
                for width in [640, 480, 1920, 1080, 800, 600, 320, 240]:
                    if total_pixels % width == 0:
                        height = total_pixels // width
                        possible_sizes.append((width, height))
                
                print(f"Possible image dimensions: {possible_sizes}")
                
                if possible_sizes:
                    width, height = possible_sizes[0]
                    mask_array = mask_array.reshape(height, width)
                    binary_mask = (mask_array > 128).astype(np.uint8)
                    print(f"Reshaped to {width}x{height}, unique values: {np.unique(binary_mask)}")
                    return binary_mask
                else:
                    print("Could not determine valid image dimensions")
                    
            except Exception as raw_error:
                print(f"Raw binary interpretation failed: {raw_error}")
                
            return None
        
        # Convert to PIL Image
        try:
            mask_image = Image.open(BytesIO(mask_bytes))
            print(f"Successfully opened PNG image: {mask_image.size}, mode: {mask_image.mode}")
        except Exception as pil_error:
            print(f"PIL Image.open error: {pil_error}")
            return None
        
        # Convert to numpy array (grayscale)
        mask_array = np.array(mask_image)
        print(f"Converted to numpy array: shape {mask_array.shape}, dtype {mask_array.dtype}")
        
        # Ensure binary mask (0 or 255)
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]  # Take first channel
            print("Converted to single channel")
        
        # Convert to binary (0 or 1)
        binary_mask = (mask_array > 128).astype(np.uint8)
        print(f"Final binary mask: shape {binary_mask.shape}, unique values: {np.unique(binary_mask)}")
        
        return binary_mask
        
    except Exception as e:
        print(f"Error decoding mask: {e}")
        import traceback
        traceback.print_exc()
        return None


def overlay_segmentation_mask(img, mask, color, alpha=0.5):
    """
    Overlay a colored segmentation mask on an image
    
    Args:
        img: OpenCV image (BGR format)
        mask: Binary mask (0s and 1s)
        color: BGR color tuple for the mask
        alpha: Transparency level (0.0 to 1.0)
    
    Returns:
        Image with overlaid mask
    """
    if mask is None:
        return img
    
    # Create colored mask
    colored_mask = np.zeros_like(img)
    colored_mask[mask == 1] = color
    
    # Blend with original image
    result = cv2.addWeighted(img, 1 - alpha, colored_mask, alpha, 0)
    
    return result


def plot_segmentation_opencv(img, segmentations_data, save_path=None):
    """
    Plot segmentation masks and bounding boxes on OpenCV image
    
    Args:
        img: OpenCV image (BGR format)
        segmentations_data: List of segmentation dictionaries with 'box_2d', 'label', and optionally 'mask'
        save_path: Optional path to save the annotated image
    
    Returns:
        Annotated OpenCV image
    """
    if not segmentations_data:
        return img
    
    img_height, img_width = img.shape[:2]
    annotated = img.copy()
    
    # Color palette for different objects
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 255),  # Orange-blue
        (128, 255, 0),  # Lime
    ]
    
    for i, seg in enumerate(segmentations_data):
        if not isinstance(seg, dict) or 'box_2d' not in seg or 'label' not in seg:
            continue
        
        # Get color for this segmentation
        color = colors[i % len(colors)]
        
        # Handle mask if present
        if 'mask' in seg and seg['mask']:
            mask = decode_segmentation_mask(seg['mask'])
            if mask is not None:
                # Resize mask to match image dimensions if needed
                if mask.shape != (img_height, img_width):
                    mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                
                # Overlay the mask
                annotated = overlay_segmentation_mask(annotated, mask, color, alpha=0.4)
        
        # Draw bounding box
        box_2d = seg['box_2d']
        y1 = int((box_2d[0] / 1000) * img_height)
        x1 = int((box_2d[1] / 1000) * img_width)
        y2 = int((box_2d[2] / 1000) * img_height)
        x2 = int((box_2d[3] / 1000) * img_width)
        
        # Draw bounding box rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label = seg['label']
        if 'confidence' in seg:
            label = f"{label} ({seg['confidence']:.2f})"
        
        # Calculate text size and background
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
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, annotated)
    
    return annotated


def plot_segmentation_pil(img, segmentations_data, save_path=None):
    """
    Plot segmentation masks and bounding boxes on PIL image
    
    Args:
        img: PIL Image
        segmentations_data: List of segmentation dictionaries
        save_path: Optional path to save the annotated image
    
    Returns:
        Annotated PIL Image
    """
    if not segmentations_data:
        return img
    
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("calibri.ttf", 20)
        except:
            font = ImageFont.load_default()
    
    # Color palette
    colors = [
        'green', 'blue', 'red', 'cyan', 'magenta', 'yellow',
        'purple', 'orange', 'lime', 'pink'
    ]
    
    for i, seg in enumerate(segmentations_data):
        if not isinstance(seg, dict) or 'box_2d' not in seg or 'label' not in seg:
            continue
        
        color = colors[i % len(colors)]
        
        # Handle mask if present (PIL implementation)
        if 'mask' in seg and seg['mask']:
            mask = decode_segmentation_mask(seg['mask'])
            if mask is not None:
                # Convert mask to PIL and overlay (simplified)
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                if mask_pil.size != (img_width, img_height):
                    mask_pil = mask_pil.resize((img_width, img_height), Image.NEAREST)
                
                # Create colored overlay
                overlay = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                # This is a simplified version - full implementation would require pixel-level processing
                # For now, just draw bounding box with transparency indication
        
        # Draw bounding box
        box_2d = seg['box_2d']
        y1 = int((box_2d[0] / 1000) * img_height)
        x1 = int((box_2d[1] / 1000) * img_width)
        y2 = int((box_2d[2] / 1000) * img_height)
        x2 = int((box_2d[3] / 1000) * img_width)
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label = seg['label']
        if 'confidence' in seg:
            label = f"{label} ({seg['confidence']:.2f})"
        
        # Draw text background
        bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 25), label, fill='white', font=font)
    
    # Save if path provided
    if save_path:
        img.save(save_path)
    
    return img


def display_image_with_segmentations(img, segmentations, save_path=None):
    """
    Unified function to display segmentations on images
    Automatically detects image format and uses appropriate plotting function
    
    Args:
        img: PIL Image or OpenCV image
        segmentations: Segmentation data
        save_path: Optional path to save annotated image
    
    Returns:
        Annotated image (same format as input)
    """
    is_opencv = isinstance(img, np.ndarray)
    
    if is_opencv:
        return plot_segmentation_opencv(img, segmentations, save_path)
    else:
        return plot_segmentation_pil(img, segmentations, save_path)