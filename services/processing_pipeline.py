from pathlib import Path

import cv2

from preprocessing.frame_extractor import FrameExtractor
from preprocessing.face_detector import FaceDetector
from preprocessing.face_cropper import FaceCropper
from preprocessing.mouth_cropper import MouthCropper

from utils.logger import logger


class ProcessingPipeline:

    def __init__(self, video_path: Path):

        self.video_path = Path(video_path)

    def run(self):

        logger.info("Starting processing pipeline...")

        metadata = self._read_metadata()

        # ----------------------------
        # Frame Extraction
        # ----------------------------

        frame_extractor = FrameExtractor(
            Path("outputs/frames")
        )

        frame_result = frame_extractor.extract(
            self.video_path
        )

        # ----------------------------
        # Face Detection
        # ----------------------------

        detector = FaceDetector()

        detection_result = detector.detect(
            frame_result["frames"]
        )

        detector.close()

        # ----------------------------
        # Face Cropping
        # ----------------------------

        face_cropper = FaceCropper(
            Path("outputs/cropped_faces")
        )

        face_result = face_cropper.crop(
            detection_result
        )

        # ----------------------------
        # Mouth Cropping
        # ----------------------------

        mouth_cropper = MouthCropper(
            Path("outputs/mouths")
        )

        mouth_result = mouth_cropper.crop(
            face_result
        )

        logger.info("Pipeline completed successfully.")

        return {

            "metadata": metadata,

            "frames": frame_result,

            "detections": detection_result,

            "faces": face_result,

            "mouths": mouth_result,

        }

    def _read_metadata(self):

        capture = cv2.VideoCapture(
            str(self.video_path)
        )

        if not capture.isOpened():

            raise ValueError(
                "Unable to open video."
            )

        width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        frame_count = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        duration = (
            round(frame_count / fps, 2)
            if fps > 0
            else 0
        )

        capture.release()

        return {

            "width": width,

            "height": height,

            "fps": round(fps, 2),

            "frame_count": frame_count,

            "duration": duration,

        }