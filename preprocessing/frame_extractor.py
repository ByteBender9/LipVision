from pathlib import Path
from typing import Dict, List

import cv2

from utils.logger import logger


class FrameExtractor:
    """
    Extracts frames from a video and saves them
    into the output frames directory.
    """

    def __init__(self, output_directory: Path):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def extract(self, video_path: Path) -> Dict:
        """
        Extract all frames from a video.

        Args:
            video_path: Path to the uploaded video.

        Returns:
            Dictionary containing extracted frame information.
        """

        video_path = Path(video_path)

        logger.info(f"Opening video: {video_path.name}")

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            logger.error("Unable to open video.")
            raise ValueError("Unable to open video.")

        frame_paths: List[Path] = []
        frame_number = 0

        logger.info("Starting frame extraction...")

        while True:

            success, frame = capture.read()

            if not success:
                break

            frame_file = (
                self.output_directory
                / f"frame_{frame_number:06d}.jpg"
            )

            cv2.imwrite(str(frame_file), frame)

            frame_paths.append(frame_file)

            frame_number += 1

        capture.release()

        logger.info(
            f"Frame extraction completed. "
            f"{frame_number} frames saved."
        )

        return {
            "success": True,
            "frame_count": frame_number,
            "frames": frame_paths,
            "output_directory": self.output_directory,
        }