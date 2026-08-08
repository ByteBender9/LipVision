"""
Inference utilities for LipVision.

Loads the trained CNN + BiLSTM model and performs
lip-reading inference on a preprocessed mouth sequence.
"""

from pathlib import Path

import torch

from training.model import LipReadingModel
from training.ctc_decoder import greedy_decode
from training.vocabulary import Vocabulary


class LipReadingInference:
    """
    Runs inference using a trained LipVision model.
    """

    def __init__(
        self,
        model_path: str | Path = "models/weights/lipreading_best.pt",
        device: torch.device | None = None,
    ) -> None:

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: "
                f"{self.model_path}"
            )

        self.device = (
            device
            if device is not None
            else self._get_device()
        )

        self.vocabulary = Vocabulary()

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
        )

        self.model = LipReadingModel(
            vocabulary_size=checkpoint[
                "vocabulary_size"
            ],
            hidden_size=checkpoint[
                "hidden_size"
            ],
            num_layers=checkpoint[
                "num_layers"
            ],
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

    @staticmethod
    def _get_device() -> torch.device:
        """
        Select the best available device.
        """

        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    @torch.no_grad()
    def predict(
        self,
        mouth_sequence: torch.Tensor,
    ) -> str:
        """
        Predict text from a mouth-frame sequence.

        Expected input:

            (frames, 1, height, width)

        Returns:
            Predicted text.
        """

        if mouth_sequence.dim() != 4:
            raise ValueError(
                "Expected mouth sequence with shape "
                "(frames, 1, height, width)."
            )

        # Add batch dimension.
        inputs = mouth_sequence.unsqueeze(0)

        inputs = inputs.to(
            self.device
        )

        logits = self.model(
            inputs
        )

        return greedy_decode(
            logits,
            self.vocabulary,
        )