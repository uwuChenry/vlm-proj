# Webcam VLM Scene Description Tool

This Python script captures images from a USB webcam and sends them to Vision Language Models (VLMs) for detailed scene description and analysis.

## 🆓 **FREE Option Available!**
- **Google Gemini Vision**: Completely FREE with API key
- **OpenAI GPT-4V**: Paid service (high quality)

## Features

- 🎥 **Webcam Integration**: Capture images from any USB webcam
- 🤖 **Multiple VLM Providers**: 
  - **Google Gemini** (FREE)
  - **OpenAI GPT-4V** (PAID)
- 💾 **Image Saving**: Automatically save captured images
- 🎯 **Custom Prompts**: Support for custom analysis prompts
- 📊 **Detailed Results**: Get comprehensive analysis results

## Prerequisites

- Python 3.7 or higher
- USB webcam connected to your computer
- API key for your chosen provider:
  - **Gemini**: Free Google API key
  - **OpenAI**: Paid OpenAI API key

## Installation

1. **Clone or download this project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

### 1. Get API Key (Choose One)

#### Option A: Google Gemini (FREE) 🆓
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

#### Option B: OpenAI GPT-4V (PAID) 💰
1. Sign up at [OpenAI](https://platform.openai.com/)
2. Create an API key with access to GPT-4V
3. Note: GPT-4V usage incurs costs based on OpenAI's pricing

### 2. Set Environment Variable

**For Google Gemini (Windows PowerShell):**
```powershell
$env:GOOGLE_API_KEY="your-google-api-key-here"
```

**For OpenAI (Windows PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-openai-api-key-here"
```

**For Linux/Mac:**
```bash
# Gemini
export GOOGLE_API_KEY="your-google-api-key-here"

# OpenAI  
export OPENAI_API_KEY="your-openai-api-key-here"
```

Alternatively, create a `.env` file in the project directory:
```
# For Gemini (FREE)
GOOGLE_API_KEY=your-google-api-key-here

# For OpenAI (PAID)
OPENAI_API_KEY=your-openai-api-key-here
```

## Usage

### Basic Usage
```bash
python webcam_vlm.py
```

### How it works:
1. **Camera Preview**: The script opens your webcam and shows a live preview
2. **Capture Image**: Press 'c' to capture an image, or 'q' to quit
3. **VLM Analysis**: The captured image is sent to GPT-4V for analysis
4. **Results**: Get a detailed description of the scene

### Example Output
```
Webcam VLM Scene Description Tool
=================================
Choose your VLM provider:
1. Google Gemini (FREE)
2. OpenAI GPT-4V (PAID)
Enter choice (1 for Gemini, 2 for OpenAI): 1
Using Google Gemini (Free) - Make sure you have GOOGLE_API_KEY set!

This tool will capture an image from your webcam and send it to the VLM for analysis.

Enter a custom prompt (or press Enter for default): 
Using default prompt: 'Describe the scene and what you see'

Camera opened successfully. Press 'c' to capture, 'q' to quit.
Image captured successfully!
Image saved to captured_image.jpg
Sending image to Google Gemini for analysis...

==================================================
ANALYSIS RESULTS
==================================================
Provider: GEMINI
Model used: gemini-1.5-flash
Image saved: captured_image.jpg

Scene Description:
--------------------
The image shows a modern home office setup with a person sitting at a wooden desk. 
The desk features a laptop computer, a coffee mug, and some notebooks. Behind the 
person, there are bookshelves filled with various books and decorative items. 
The lighting appears to be natural daylight coming from a window on the left side 
of the frame, creating a warm and productive workspace atmosphere.
==================================================
```

## Configuration Options

### Camera Selection
If you have multiple cameras, you can modify the camera index in the script:
```python
result = webcam_vlm.capture_and_analyze(
    camera_index=1,  # Change to 1, 2, etc. for different cameras
    prompt=custom_prompt,
    save_image=True
)
```

### VLM Provider Selection
The script now supports both providers:
```python
# Use Gemini (FREE)
webcam_vlm = WebcamVLM(provider="gemini")

# Use OpenAI (PAID)
webcam_vlm = WebcamVLM(provider="openai")
```

### Custom Analysis Prompts
You can provide custom prompts for specific analysis needs:
- "Count the number of people in the image"
- "Describe the emotions and expressions you see"
- "Identify any text or signs visible in the scene"
- "Analyze the lighting and composition of this photo"

### Model Configuration
The script uses different models based on provider:
- **Gemini**: `gemini-1.5-flash` (fast and free)
- **OpenAI**: `gpt-4o` (high quality, paid)

## File Structure
```
vlm proj/
├── webcam_vlm.py          # Main script (supports Gemini + OpenAI)
├── requirements.txt       # Python dependencies  
├── README.md             # This file
├── .env.example          # API key template
├── captured_image.jpg    # Saved images (created when running)
└── .venv/               # Virtual environment (created automatically)
```

## Troubleshooting

### Camera Issues
- **Camera not found**: Make sure your webcam is connected and not being used by another application
- **Permission denied**: On some systems, you may need to grant camera permissions
- **Multiple cameras**: Try different camera indices (0, 1, 2, etc.)

### API Issues
- **Invalid API key**: Verify your API key is correct
  - **Gemini**: Check at [Google AI Studio](https://makersuite.google.com/app/apikey)
  - **OpenAI**: Check at [OpenAI Platform](https://platform.openai.com/api-keys)
- **Rate limiting**: Both services have rate limits; wait a moment and try again
- **Insufficient credits**: 
  - **Gemini**: Check quota at Google AI Studio
  - **OpenAI**: Check account balance

### Installation Issues
- **OpenCV installation**: On some systems, you might need `python-opencv` instead of `opencv-python`
- **Virtual environment**: Consider using a virtual environment to avoid package conflicts

## Cost Considerations

### Google Gemini (FREE) 🆓
- **Free tier**: Generous free quota for personal use
- **Rate limits**: 15 requests per minute
- **No credit card required**

### OpenAI GPT-4V (PAID) 💰
OpenAI charges based on:
- **Image analysis**: Per image processed
- **Token usage**: Based on input prompt and output length

Current pricing at [OpenAI's pricing page](https://openai.com/pricing).

## Advanced Usage

### Programmatic Usage
You can also use the `WebcamVLM` class in your own scripts:

```python
from webcam_vlm import WebcamVLM

# Initialize with Gemini (FREE)
vlm = WebcamVLM(provider="gemini", api_key="your-google-api-key")

# Or with OpenAI (PAID)
vlm = WebcamVLM(provider="openai", api_key="your-openai-api-key")

# Capture and analyze
result = vlm.capture_and_analyze(
    camera_index=0,
    prompt="What objects do you see on the desk?",
    save_image=True
)

print(result["analysis"])
```

### Batch Processing
For processing multiple images, you can modify the script to capture several images in sequence or process existing image files.

## Contributing

Feel free to contribute improvements:
- Add support for other VLM APIs (Anthropic Claude, etc.)
- Implement video analysis capabilities
- Add GUI interface
- Improve error handling and user experience

## Quick Start (TL;DR)

1. **Get a FREE Google API key**: https://makersuite.google.com/app/apikey
2. **Set environment variable**: 
   ```powershell
   $env:GOOGLE_API_KEY="your-key-here"
   ```
3. **Run the script**: 
   ```bash
   python webcam_vlm.py
   ```
4. **Choose option 1** (Gemini - FREE)
5. **Press 'c'** to capture and analyze!

## License

This project is provided as-is for educational and personal use. Please respect OpenAI's terms of service when using their API.