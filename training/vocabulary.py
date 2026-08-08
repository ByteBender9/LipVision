"""
Vocabulary utilities for LipVision CTC training.
"""

import string


class Vocabulary:
    """
    Converts text characters to integer IDs and back.

    The vocabulary contains:
        - blank token for CTC
        - space
        - lowercase letters
        - digits
    """

    def __init__(self) -> None:

        self.blank_token = "<blank>"

        characters = (
            string.ascii_lowercase
            + string.digits
            + " "
        )

        self.tokens = [
            self.blank_token,
            *characters,
        ]

        self.char_to_id = {
            char: index
            for index, char in enumerate(self.tokens)
        }

        self.id_to_char = {
            index: char
            for char, index in self.char_to_id.items()
        }

    @property
    def size(self) -> int:
        """Return the vocabulary size."""

        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        """
        Convert text into character IDs.
        """

        text = text.lower().strip()

        unknown_characters = [
            char
            for char in text
            if char not in self.char_to_id
        ]

        if unknown_characters:
            raise ValueError(
                "Unknown characters found: "
                f"{unknown_characters}"
            )

        return [
            self.char_to_id[char]
            for char in text
        ]

    def decode(
        self,
        token_ids: list[int],
        remove_blank: bool = True,
    ) -> str:
        """
        Convert character IDs back into text.
        """

        characters = []

        for token_id in token_ids:

            if token_id not in self.id_to_char:
                raise ValueError(
                    f"Unknown token ID: {token_id}"
                )

            if (
                remove_blank
                and token_id
                == self.char_to_id[self.blank_token]
            ):
                continue

            characters.append(
                self.id_to_char[token_id]
            )

        return "".join(characters)

    def __len__(self) -> int:
        return self.size