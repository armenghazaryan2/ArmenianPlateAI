import cv2


image = cv2.imread("data/test_images/test.jpg")


if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")
    print("Image size:", image.shape)