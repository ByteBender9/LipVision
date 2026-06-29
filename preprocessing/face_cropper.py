import cv2
import os


class FaceCropper:

    def __init__(self):
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

    def crop_faces(self, input_folder, output_folder):

        os.makedirs(output_folder, exist_ok=True)

        count = 0

        for image_name in sorted(os.listdir(input_folder)):

            image_path = os.path.join(input_folder, image_name)

            image = cv2.imread(image_path)

            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5
            )

            if len(faces) == 0:
                continue

            x, y, w, h = faces[0]

            face = image[y:y+h, x:x+w]

            cv2.imwrite(
                os.path.join(output_folder, image_name),
                face
            )

            count += 1

        return count