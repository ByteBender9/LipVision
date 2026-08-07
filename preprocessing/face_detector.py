from pathlib import Path

import cv2
import mediapipe as mp

from models.detection import (
    BoundingBox,
    DetectionResults,
    FaceDetection,
)

from utils.logger import logger


class FaceDetector:
    """
    Detects faces and facial landmarks using
    MediaPipe Face Mesh.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(
        self,
        frame_paths: list[Path],
    ) -> DetectionResults:

        logger.info("Starting face detection...")

        detections: list[FaceDetection] = []

        for frame_path in frame_paths:

            image = cv2.imread(str(frame_path))

            if image is None:
                continue

            height, width = image.shape[:2]

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            results = self.face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                continue

            landmarks = results.multi_face_landmarks[0]

            xs = []
            ys = []

            for landmark in landmarks.landmark:

                xs.append(
                    int(landmark.x * width)
                )

                ys.append(
                    int(landmark.y * height)
                )

            padding = 20

            x_min = max(min(xs) - padding, 0)

            y_min = max(min(ys) - padding, 0)

            x_max = min(max(xs) + padding, width)

            y_max = min(max(ys) + padding, height)

            bbox = BoundingBox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )

            detections.append(

                FaceDetection(

                    frame_path=frame_path,

                    image_width=width,

                    image_height=height,

                    bounding_box=bbox,

                    landmarks=landmarks,

                )

            )

        logger.info(
            "Detected %d face(s).",
            len(detections),
        )

        return DetectionResults(

            success=True,

            detections=detections,

        )

    def close(self):

        self.face_mesh.close()