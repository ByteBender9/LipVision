"""
LipVision CNN + BiLSTM + CTC model.
"""

import torch
import torch.nn as nn


class LipReadingModel(nn.Module):
    """
    CNN + BiLSTM model for visual speech recognition.

    Input:
        (batch, frames, 1, 96, 96)

    Output:
        (frames, batch, vocabulary_size)
    """

    def __init__(
        self,
        vocabulary_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
    ) -> None:

        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 96x96 → 48x48 → 24x24 → 12x12
        self.feature_size = 128 * 12 * 12

        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(
            hidden_size * 2,
            vocabulary_size,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Tensor with shape:
            (batch, frames, 1, 96, 96)

        Returns
        -------
        torch.Tensor
            Logits with shape:
            (frames, batch, vocabulary_size)
        """

        batch_size, frames, channels, height, width = x.shape

        # Process every video frame through CNN.
        x = x.reshape(
            batch_size * frames,
            channels,
            height,
            width,
        )

        x = self.cnn(x)

        # Flatten CNN features.
        x = x.reshape(
            batch_size,
            frames,
            -1,
        )

        # Temporal modelling.
        x, _ = self.lstm(x)

        # Character/token probabilities.
        x = self.classifier(x)

        # CTC expects:
        # (sequence_length, batch, classes)
        x = x.permute(1, 0, 2)

        return x