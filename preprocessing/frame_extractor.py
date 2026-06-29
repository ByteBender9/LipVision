import cv2
import os


class FrameExtractor:

    def __init__(self, video_path, output_folder):

        self.video_path = video_path
        self.output_folder = output_folder

        os.makedirs(output_folder, exist_ok=True)

    def extract(self):

        cap = cv2.VideoCapture(self.video_path)

        count = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            filename = os.path.join(
                self.output_folder,
                f"frame_{count:04d}.jpg"
            )

            cv2.imwrite(filename, frame)

            count += 1

        cap.release()

        return count