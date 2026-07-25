from ultralytics import YOLO

# Load trained Armenian plate model
model = YOLO("runs/detect/train-3/weights/best.pt")

# Test image
image_path = "data/images/train/7434404e-001.jpg"

# Run AI detection
results = model(image_path)

# Show result
for result in results:
    result.show()

    print("Detected boxes:")
    print(result.boxes)