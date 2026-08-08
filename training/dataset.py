"""
GRID dataset loader for LipVision.

Loads GRID videos, extracts mouth regions using MediaPipe Face Mesh,
and returns video sequences with their corresponding transcripts.
"""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch.utils.data import Dataset

from training.grid_labels import decode_grid_file


class GRIDDataset(Dataset):
    """
    PyTorch Dataset for the GRID corpus.

    Expected structure:

        GRID/
        └── video/
            └── s1/
                └── s1/
                    ├── prwq3s.mpg
                    ├── pbib8p.mpg
                    └── ...

    The transcript is decoded automatically from each filename.
    """

    MOUTH_INDICES = [
        61, 146, 91, 181, 84, 17,
        314, 405, 321, 375, 291,
        78, 95, 88, 178, 87,
        14, 317, 402, 318, 324, 308,
    ]

    def __init__(
        self,
        video_directory: Path | str,
        image_size: tuple[int, int] = (96, 96),
        max_frames: int | None = None,
    ) -> None:

        self.video_directory = Path(video_directory)

        self.image_size = image_size
        self.max_frames = max_frames

        if not self.video_directory.exists():
            raise FileNotFoundError(
                f"GRID video directory not found: "
                f"{self.video_directory}"
            )

        self.video_paths = sorted(
            self.video_directory.glob("*.mpg")
        )

        if not self.video_paths:
            raise ValueError(
                f"No GRID videos found in "
                f"{self.video_directory}"
            )

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, str]:

        video_path = self.video_paths[index]

        sequence = self._extract_mouth_sequence(
            video_path
        )

        transcript = decode_grid_file(
            video_path
        )

        return sequence, transcript

    def _extract_mouth_sequence(
        self,
        video_path: Path,
    ) -> torch.Tensor:
        """
        Extract mouth crops from every frame of a video.
        """

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise ValueError(
                f"Unable to open video: {video_path}"
            )

        frames: list[np.ndarray] = []

        while True:

            success, frame = capture.read()

            if not success:
                break

            mouth = self._extract_mouth(frame)

            if mouth is None:
                continue

            mouth = cv2.resize(
                mouth,
                self.image_size,
                interpolation=cv2.INTER_AREA,
            )

            mouth = mouth.astype(
                np.float32
            ) / 255.0

            frames.append(mouth)

            if (
                self.max_frames is not None
                and len(frames) >= self.max_frames
            ):
                break

        capture.release()

        if not frames:
            raise ValueError(
                f"No mouth frames detected in: "
                f"{video_path}"
            )

        sequence = np.stack(
            frames,
            axis=0,
        )

        # (frames, height, width)
        # →
        # (frames, 1, height, width)

        sequence = np.expand_dims(
            sequence,
            axis=1,
        )

        return torch.from_numpy(
            sequence
        ).float()

    def _extract_mouth(
        self,
        frame: np.ndarray,
    ) -> np.ndarray | None:
        """
        Detect the face and extract the mouth region.
        """

        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        landmarks = (
            result.multi_face_landmarks[0]
        )

        xs = []
        ys = []

        for index in self.MOUTH_INDICES:

            landmark = landmarks.landmark[index]

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            xs.append(x)
            ys.append(y)

        padding_x = 15
        padding_y = 15

        x_min = max(
            min(xs) - padding_x,
            0,
        )

        y_min = max(
            min(ys) - padding_y,
            0,
        )

        x_max = min(
            max(xs) + padding_x,
            width,
        )

        y_max = min(
            max(ys) + padding_y,
            height,
        )

        mouth = frame[
            y_min:y_max,
            x_min:x_max,
        ]

        if mouth.size == 0:
            return None

        return cv2.cvtColor(
            mouth,
            cv2.COLOR_BGR2GRAY,
        )

    def close(self) -> None:
        """
        Release MediaPipe resources.
        """

        self.face_mesh.close()