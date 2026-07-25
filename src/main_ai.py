import cv2

from detection.plate_detector import PlateDetector
from ocr.read_plate import read_plate


image_path = "data/images/val/test.jpg"


# Load image
image = cv2.imread(image_path)


# Detect plate
detector = PlateDetector()

boxes = detector.detect(image_path)


if len(boxes) == 0:

    print("No plate detected")

else:

    for box in boxes:

        coordinates = box.xyxy[0]

        x1, y1, x2, y2 = map(int, coordinates)

        print(
            "Plate location:",
            x1,
            y1,
            x2,
            y2
        )


        # Crop plate
        plate_image = image[
            y1:y2,
            x1:x2
        ]


        # OCR
        plate_text = read_plate(
            plate_image
        )


        print(
            "Detected plate:",
            plate_text
        )