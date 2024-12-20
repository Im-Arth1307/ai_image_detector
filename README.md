# AI Image Detector

A Python-based project to detect AI-generated or enhanced images using deep learning techniques.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your test images in the `test_images` folder
2. Run the detector:
```bash
python detector.py
```

## Features

- Uses a pre-trained CNN model to detect AI-generated images
- Provides confidence scores for predictions
- Supports common image formats (jpg, png, etc.)

## Note
This is a basic implementation and may need fine-tuning based on specific use cases.

## Our Approach
We're using transfer learning - leveraging ResNet50's learned features
Instead of training from scratch, we use its ability to extract features
The features from AI images often have different statistical properties
For example:
More uniform feature distributions
Different standard deviations in feature patterns
Unusual correlations between features