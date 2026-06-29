import cv2
import os


class FaceDetector:

    def __init__(self):
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_faces(self, input_folder, output_folder):

        os.makedirs(output_folder, exist_ok=True)

        total = 0

        for image_name in sorted(os.listdir(input_folder)):

            image_path = os.path.join(input_folder, image_name)

            image = cv2.imread(image_path)

            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            for (x, y, w, h) in faces:

                cv2.rectangle(
                    image,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

            cv2.imwrite(
                os.path.join(output_folder, image_name),
                image
            )

            total += 1

        return total