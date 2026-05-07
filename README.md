# Diabetic Retinopathy Detection System

Flask web application for diabetic retinopathy screening from retinal fundus images using a MobileNetV2 + GCN hybrid model, Grad-CAM visualizations, and doctor-focused case management.

## Why This Project

- Supports early diabetic retinopathy triage from retinal images.
- Provides explainability with heatmaps and optional AI-generated clinical explanations.
- Includes role-based access so doctors can manage analyses and review history.

## Core Features

- Hybrid deep learning architecture (MobileNetV2 + graph convolution).
- Image enhancement options during preprocessing.
- Single-eye and dual-eye analysis workflows.
- Grad-CAM heatmap generation for visual explainability.
- Case history, export endpoints, and label feedback workflow.
- Optional xAI (Grok) integration for patient-friendly explanation text.

## Tech Stack

- Python, Flask, SQLAlchemy, SQLite
- PyTorch, TorchVision, OpenCV, Pillow
- FPDF for PDF exports

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open:
   - `http://127.0.0.1:5051`

## Environment Variables

Optional settings you can configure:

- `PORT`: app port (default `5051`)
- `HOST`: bind host (default `0.0.0.0`)
- `FLASK_DEBUG`: debug mode (`1`/`true`/`yes`/`on` enables; default enabled)
- `MODEL_WEIGHTS_PATH`: absolute or relative path to model weights file
- `XAI_API_KEY`: key for xAI API integration

PowerShell example:

```powershell
$env:PORT="5051"
$env:FLASK_DEBUG="1"
$env:MODEL_WEIGHTS_PATH="C:\path\to\model_weights.pth"
python app.py
```

## Model Weights

- By default, the app looks for `model_weights.pth` in the project root.
- If the file is missing, the app starts with initialized weights so UI workflows still run.
- For meaningful medical predictions, train on a retinal dataset and provide trained weights.

## User Roles

- `user`: basic access
- `admin` (doctor): dashboard and clinical workflows

Register as doctor/admin through the registration form role field.

## Testing

Run smoke tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

These tests validate that key public routes load successfully.

## Dataset

The model can be trained with the [APTOS 2019 Blindness Detection dataset](https://www.kaggle.com/c/aptos2019-blindness-detection).

## Important Note

This project is intended for educational and engineering demonstration use. It is not a certified medical device and should not be used as the sole basis for clinical decisions.
