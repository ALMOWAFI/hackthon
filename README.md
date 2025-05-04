# Math Homework Analyzer

A Python-based system that analyzes elementary-level math homework from scanned images, providing detailed feedback and visual annotations.

## Features

- OCR for handwritten math expressions (including symbols like ×, ÷, fractions)
- Per-question segmentation and spatial mapping
- Step-by-step error detection (procedural, conceptual, calculation)
- Visual annotations (red circles for mistakes, green checks for correct steps)
- Detailed JSON reports with analysis and recommendations
- Local storage of annotated images and feedback

## Setup

1. Install Tesseract OCR:
   - Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Google Gemini API:
   - Create a Google Cloud project
   - Enable the Gemini API
   - Set your API key in config.py

## Usage

```python
from math_analyzer import MathHomeworkAnalyzer

analyzer = MathHomeworkAnalyzer()
results = analyzer.analyze_homework("path/to/homework_image.jpg")
```

## Output

The system generates:
- Annotated image with visual feedback
- feedback.txt with detailed explanations
- analysis_results.json with structured data

## Project Structure

```
math_analyzer/
├── __init__.py
├── ocr.py           # OCR processing
├── segmentation.py  # Image segmentation
├── analysis.py      # Math analysis
├── visualization.py # Visual annotations
├── feedback.py      # Feedback generation
└── utils.py         # Utility functions
``` 