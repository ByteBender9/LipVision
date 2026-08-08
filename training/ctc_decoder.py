"""
CTC decoding utilities for LipVision.
"""

import torch

from training.vocabulary import Vocabulary


def greedy_decode(
    logits: torch.Tensor,
    vocabulary: Vocabulary,
) -> str:
    """
    Decode CTC model output using greedy decoding.
    """

    if logits.dim() == 3:

        if logits.shape[1] != 1:
            raise ValueError(
                "greedy_decode expects a single "
                "sample when logits are 3-dimensional."
            )

        logits = logits[:, 0, :]

    elif logits.dim() != 2:

        raise ValueError(
            "Expected logits with shape "
            "(sequence, classes) or "
            "(sequence, batch, classes)."
        )

    token_ids = torch.argmax(
        logits,
        dim=-1,
    ).tolist()

    blank_id = vocabulary.char_to_id[
        vocabulary.blank_token
    ]

    decoded_ids = []

    previous_id = None

    for token_id in token_ids:

        # CTC blank
        if token_id == blank_id:
            previous_id = token_id
            continue

        # Collapse repeated tokens
        if token_id == previous_id:
            continue

        decoded_ids.append(token_id)

        previous_id = token_id

    return vocabulary.decode(
        decoded_ids,
        remove_blank=True,
    )


def greedy_decode_batch(
    logits: torch.Tensor,
    vocabulary: Vocabulary,
) -> list[str]:
    """
    Decode a batch of CTC predictions.
    """

    if logits.dim() != 3:
        raise ValueError(
            "Expected logits with shape "
            "(sequence, batch, classes)."
        )

    predictions = torch.argmax(
        logits,
        dim=-1,
    )

    blank_id = vocabulary.char_to_id[
        vocabulary.blank_token
    ]

    results = []

    for batch_index in range(
        predictions.shape[1]
    ):

        token_ids = predictions[
            :, batch_index
        ].tolist()

        decoded_ids = []

        previous_id = None

        for token_id in token_ids:

            if token_id == blank_id:
                previous_id = token_id
                continue

            if token_id == previous_id:
                continue

            decoded_ids.append(token_id)

            previous_id = token_id

        text = vocabulary.decode(
            decoded_ids,
            remove_blank=True,
        )

        results.append(text)

    return results