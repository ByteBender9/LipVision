"""
Detection data models for LipVision.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import mediapipe as mp


@dataclass(slots=True)
class BoundingBox:
    """
    Bounding box around a detected face.
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass(slots=True)
class FaceDetection:
    """
    Information about one detected face.
    """

    frame_path: Path

    image_width: int
    image_height: int

    bounding_box: BoundingBox

    landmarks: Any


@dataclass(slots=True)
class DetectionResults:
    """
    Collection of detected faces.
    """

    success: bool

    detections: List[FaceDetection]

    @property
    def count(self) -> int:
        return len(self.detections)