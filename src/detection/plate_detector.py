from ultralytics import YOLO


class PlateDetector:

    def __init__(self):
        self.model = YOLO(
            "runs/detect/train-3/weights/best.pt"
        )


    def detect(self, image_path):

        results = self.model(image_path)

        boxes = results[0].boxes

        return boxes