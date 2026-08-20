from src.detection.plate_detector import PlateDetector

IMAGE_PATH = "data/images/train/7434404e-001.jpg"

detector = PlateDetector()

for confidence in [0.40, 0.20, 0.10, 0.05, 0.01]:

    print("\n" + "=" * 50)
    print(f"Testing confidence: {confidence}")
    print("=" * 50)

    results = detector.model(
        IMAGE_PATH,
        conf=confidence
    )

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("No detections.")
        continue

    print(f"Number of detections: {len(boxes)}")

    for i, box in enumerate(boxes):
        print(
            f"Detection {i + 1}: "
            f"confidence={float(box.conf[0]):.6f}, "
            f"box={box.xyxy[0].tolist()}"
        )