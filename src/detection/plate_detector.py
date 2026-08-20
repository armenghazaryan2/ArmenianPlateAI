from ultralytics import YOLO

from src.config import MODEL_PATH, DETECTION_CONFIDENCE


class PlateDetector:

    def __init__(
        self,
        model_path=MODEL_PATH,
        confidence=DETECTION_CONFIDENCE
    ):
        self.model = YOLO(str(model_path))
        self.confidence = confidence

    def detect(self, image_path):

        results = self.model(
            image_path,
            conf=self.confidence
        )

        boxes = results[0].boxes

        # No detections
        if boxes is None or len(boxes) == 0:
            return None

        # Get confidence scores
        confidences = boxes.conf

        # Find highest-confidence detection
        best_index = int(confidences.argmax())

        best_box = boxes.xyxy[best_index].tolist()
        best_conf = float(confidences[best_index])

        return {
            "box": best_box,
            "confidence": best_conf
        }