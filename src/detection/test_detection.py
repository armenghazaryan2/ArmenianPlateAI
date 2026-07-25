from ultralytics import YOLO


model = YOLO("yolov8n.pt")

image_path = "data/test_images/test.jpg"

results = model(image_path)

for result in results:
    print(result)