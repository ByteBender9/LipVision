from pathlib import Path

import cv2

from models.detection import DetectionResults

from utils.logger import logger


class FaceCropper:
    """
    Crops detected faces from extracted frames.
    """

    def __init__(self, output_directory: Path):

        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def crop(
        self,
        detection_results: DetectionResults,
    ):

        logger.info("Starting face cropping...")

        cropped_faces = []

        for detection in detection_results.detections:

            image = cv2.imread(
                str(detection.frame_path)
            )

            if image is None:
                continue

            box = detection.bounding_box

            face = image[
                box.y_min:box.y_max,
                box.x_min:box.x_max,
            ]

            if face.size == 0:
                continue

            output_path = (
                self.output_directory
                / detection.frame_path.name
            )

            cv2.imwrite(
                str(output_path),
                face,
            )

            cropped_faces.append(

                {
                    "frame_path": detection.frame_path,

                    "face_path": output_path,

                    "bounding_box": box,

                    "landmarks": detection.landmarks,

                }

            )

        logger.info(
            "Saved %d cropped faces.",
            len(cropped_faces),
        )

        return {

            "success": True,

            "count": len(cropped_faces),

            "faces": cropped_faces,

        }