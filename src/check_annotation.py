import cv2
from pathlib import Path

image_path = Path("data/images/train/7434404e-001.jpg")
label_path = Path("data/labels/train/7434404e-001.txt")

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

height, width = image.shape[:2]

with open(label_path, "r") as f:
    line = f.readline().strip()

class_id, x_center, y_center, box_width, box_height = map(
    float, line.split()
)

x_center *= width
y_center *= height
box_width *= width
box_height *= height

x1 = int(x_center - box_width / 2)
y1 = int(y_center - box_height / 2)
x2 = int(x_center + box_width / 2)
y2 = int(y_center + box_height / 2)

x1 = max(0, min(x1, width - 1))
y1 = max(0, min(y1, height - 1))
x2 = max(0, min(x2, width - 1))
y2 = max(0, min(y2, height - 1))

cv2.rectangle(
    image,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    3
)

output_path = Path("annotation_check.jpg")

cv2.imwrite(str(output_path), image)

print(f"Image size: {width} x {height}")
print(f"Bounding box: ({x1}, {y1}) -> ({x2}, {y2})")
print(f"Saved to: {output_path}")