from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# YOLO model
MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "train-3"
    / "weights"
    / "best.pt"
)

# Detection
DETECTION_CONFIDENCE = 0.4

# OCR
OCR_CHAR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Default image
DEFAULT_IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "val"
    / "test.jpg"
)