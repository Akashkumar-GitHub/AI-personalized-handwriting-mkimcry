import cv2
import numpy as np
import os
from pathlib import Path

class HandwritingPreprocessor:

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.gray = None
        self.binary = None

    def load_image(self):
        self.image = cv2.imread(self.image_path)

        if self.image is None:
            raise ValueError("Image not found!")

        print("[INFO] Image loaded successfully")

    def convert_to_grayscale(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print("[INFO] Converted to grayscale")

    def remove_noise(self):
        self.gray = cv2.GaussianBlur(self.gray, (5, 5), 0)
        print("[INFO] Noise removed")

    def threshold_image(self):
        _, self.binary = cv2.threshold(
            self.gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        print("[INFO] Binary threshold applied")

    def deskew_image(self):

        coords = np.column_stack(np.where(self.binary > 0))
        if coords.size == 0:
            print("[WARN] No foreground pixels found; skipping deskew")
            return

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = self.binary.shape[:2]

        center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        self.binary = cv2.warpAffine(
            self.binary,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        print(f"[INFO] Deskewed image by {angle:.2f} degrees")

    def detect_text_contours(self):

        contours, _ = cv2.findContours(
            self.binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        output = self.image.copy()

        for contour in contours:

            x, y, w, h = cv2.boundingRect(contour)

            if w > 30 and h > 10:
                cv2.rectangle(
                    output,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

        print("[INFO] Text contours detected")

        return output

    def save_processed_image(self, output_path):

        cv2.imwrite(output_path, self.binary)

        print(f"[INFO] Processed image saved at: {output_path}")

    def process(self):

        self.load_image()

        self.convert_to_grayscale()

        self.remove_noise()

        self.threshold_image()

        self.deskew_image()

        contour_output = self.detect_text_contours()

        return contour_output


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent

    IMAGE_PATH = BASE_DIR / "sample_handwriting.jpg"

    OUTPUT_PATH = BASE_DIR / "processed_output.png"

    processor = HandwritingPreprocessor(str(IMAGE_PATH))

    contour_image = processor.process()

    processor.save_processed_image(OUTPUT_PATH)

    cv2.imshow("Detected Text Regions", contour_image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
