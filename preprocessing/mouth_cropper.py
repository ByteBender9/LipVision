from pathlib import Path

import cv2

from utils.logger import logger


class MouthCropper:
    """
    Crops the mouth region from cropped face images.
    """

    # MediaPipe Face Mesh outer + inner lip landmarks
    MOUTH_INDICES = [
        61, 146, 91, 181, 84, 17,
        314, 405, 321, 375, 291,
        78, 95, 88, 178, 87,
        14, 317, 402, 318, 324, 308
    ]

    def __init__(self, output_directory: Path):

        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def crop(self, face_results):

        logger.info("Starting mouth cropping...")

        mouths = []

        for face in face_results["faces"]:

            image = cv2.imread(str(face["frame_path"]))

            if image is None:
                continue

            box = face["bounding_box"]

            landmarks = face["landmarks"]

            xs = []
            ys = []

            for index in self.MOUTH_INDICES:

                landmark = landmarks.landmark[index]

                x = int(
                    landmark.x * image.shape[1]
                )

                y = int(
                    landmark.y * image.shape[0]
                )

                # Convert original-frame coordinates
                # into face-crop coordinates

                x -= box.x_min
                y -= box.y_min

                xs.append(x)
                ys.append(y)

            face_image = cv2.imread(
                str(face["face_path"])
            )

            if face_image is None:
                continue

            h, w = face_image.shape[:2]

            padding = 15

            x_min = max(min(xs) - padding, 0)
            y_min = max(min(ys) - padding, 0)

            x_max = min(max(xs) + padding, w)
            y_max = min(max(ys) + padding, h)

            mouth = face_image[
                y_min:y_max,
                x_min:x_max,
            ]

            if mouth.size == 0:
                continue

            output_path = (
                self.output_directory
                / face["face_path"].name
            )

            cv2.imwrite(
                str(output_path),
                mouth,
            )

            mouths.append(
                {
                    "mouth_path": output_path,
                    "frame_path": face["frame_path"],
                }
            )

        logger.info(
            "Saved %d mouth crops.",
            len(mouths),
        )

        return {

            "success": True,

            "count": len(mouths),

            "mouths": mouths,

        }