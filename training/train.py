"""
Training script for the LipVision CNN + BiLSTM + CTC model.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

from training.dataset import GRIDDataset
from training.model import LipReadingModel
from training.vocabulary import Vocabulary


# --------------------------------------------------
# Configuration
# --------------------------------------------------

GRID_VIDEO_DIRECTORY = Path(
    "/Users/macm1/Desktop/GRID/video/s1/s1"
)

MODEL_DIRECTORY = Path("models/weights")

IMAGE_SIZE = (96, 96)

BATCH_SIZE = 2

EPOCHS = 5

LEARNING_RATE = 1e-4

MAX_FRAMES = 75

TRAIN_SPLIT = 0.9

RANDOM_SEED = 42


# --------------------------------------------------
# Device
# --------------------------------------------------

def get_device() -> torch.device:
    """
    Select the best available device.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# --------------------------------------------------
# Batch preparation
# --------------------------------------------------

def collate_batch(batch):
    """
    Prepare a batch containing variable-length video sequences.
    """

    sequences, transcripts = zip(*batch)

    sequence_lengths = torch.tensor(
        [
            sequence.shape[0]
            for sequence in sequences
        ],
        dtype=torch.long,
    )

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0,
    )

    vocabulary = collate_batch.vocabulary

    encoded_targets = [
        torch.tensor(
            vocabulary.encode(transcript),
            dtype=torch.long,
        )
        for transcript in transcripts
    ]

    target_lengths = torch.tensor(
        [
            target.numel()
            for target in encoded_targets
        ],
        dtype=torch.long,
    )

    targets = torch.cat(
        encoded_targets
    )

    return (
        padded_sequences,
        targets,
        sequence_lengths,
        target_lengths,
    )


# --------------------------------------------------
# Training
# --------------------------------------------------

def train() -> None:
    """
    Train the LipVision model.
    """

    device = get_device()

    print(
        f"Using device: {device}"
    )

    vocabulary = Vocabulary()

    collate_batch.vocabulary = vocabulary

    print(
        f"Vocabulary size: {vocabulary.size}"
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = GRIDDataset(
        video_directory=GRID_VIDEO_DIRECTORY,
        image_size=IMAGE_SIZE,
        max_frames=MAX_FRAMES,
    )

    train_size = int(
        len(dataset) * TRAIN_SPLIT
    )

    validation_size = (
        len(dataset) - train_size
    )

    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(validation_dataset)}"
    )

    # --------------------------------------------------
    # Data loaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = LipReadingModel(
        vocabulary_size=vocabulary.size,
        hidden_size=256,
        num_layers=2,
    )

    model = model.to(device)

    # --------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------

    criterion = nn.CTCLoss(
        blank=0,
        zero_infinity=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Checkpoint / Resume
    # --------------------------------------------------

    best_validation_loss = float("inf")
    start_epoch = 0

    checkpoint_path = (
        MODEL_DIRECTORY
        / "lipreading_best.pt"
    )

    if checkpoint_path.exists():

        print(
            f"Loading checkpoint: "
            f"{checkpoint_path}"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # The Epoch-2 checkpoint was created
        # before optimizer state was added.
        if "optimizer_state_dict" in checkpoint:

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        else:

            print(
                "Optimizer state not found. "
                "Resuming model weights with "
                "a fresh optimizer."
            )

        start_epoch = checkpoint["epoch"]

        best_validation_loss = (
            checkpoint["validation_loss"]
        )

        print(
            f"Resuming from epoch "
            f"{start_epoch + 1}"
        )

    # --------------------------------------------------
    # Epochs
    # --------------------------------------------------

    for epoch in range(
        start_epoch,
        EPOCHS,
    ):

        model.train()

        training_loss = 0.0

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):

            (
                inputs,
                targets,
                input_lengths,
                target_lengths,
            ) = batch

            inputs = inputs.to(device)

            targets = targets.to(device)

            optimizer.zero_grad()

            logits = model(inputs)

            log_probs = torch.log_softmax(
                logits,
                dim=2,
            )

            # CTC is not implemented on MPS.
            # Run CTC loss on CPU while keeping
            # the model itself on MPS.
            loss = criterion(
                log_probs.cpu(),
                targets.cpu(),
                input_lengths.cpu(),
                target_lengths.cpu(),
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            training_loss += loss.item()

            if batch_index % 10 == 0:

                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"| Batch {batch_index}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        average_training_loss = (
            training_loss
            / len(train_loader)
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for batch in validation_loader:

                (
                    inputs,
                    targets,
                    input_lengths,
                    target_lengths,
                ) = batch

                inputs = inputs.to(device)

                targets = targets.to(device)

                logits = model(inputs)

                log_probs = torch.log_softmax(
                    logits,
                    dim=2,
                )

                loss = criterion(
                    log_probs.cpu(),
                    targets.cpu(),
                    input_lengths.cpu(),
                    target_lengths.cpu(),
                )

                validation_loss += loss.item()

        average_validation_loss = (
            validation_loss
            / len(validation_loader)
        )

        print()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} completed"
        )

        print(
            f"Training Loss:   "
            f"{average_training_loss:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{average_validation_loss:.4f}"
        )

        print()

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if (
            average_validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                average_validation_loss
            )

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "vocabulary_size": (
                        vocabulary.size
                    ),
                    "hidden_size": 256,
                    "num_layers": 2,
                    "image_size": IMAGE_SIZE,
                    "epoch": epoch + 1,
                    "validation_loss": (
                        average_validation_loss
                    ),
                },
                checkpoint_path,
            )

            print(
                f"Saved best model: "
                f"{checkpoint_path}"
            )

    dataset.close()

    print(
        "Training completed."
    )


if __name__ == "__main__":
    train()