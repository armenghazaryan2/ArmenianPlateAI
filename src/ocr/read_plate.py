import pytesseract
import cv2


def read_plate(image):

    text = pytesseract.image_to_string(
        image,
        config="--psm 7"
    )

    text = text.strip()

    return text