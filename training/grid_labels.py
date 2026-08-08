"""
GRID Corpus utterance-code decoder.

GRID filenames contain a six-character utterance code:

    C C P L D A

    C = command
    C = color
    P = preposition
    L = letter
    D = digit
    A = adverb
"""

from pathlib import Path


COMMANDS = {
    "b": "bin",
    "l": "lay",
    "p": "place",
    "s": "set",
}

COLORS = {
    "b": "blue",
    "g": "green",
    "r": "red",
    "w": "white",
}

PREPOSITIONS = {
    "a": "at",
    "b": "by",
    "i": "in",
    "w": "with",
}

ADVERBS = {
    "a": "again",
    "n": "now",
    "p": "please",
    "s": "soon",
}


def decode_grid_code(code: str) -> str:
    """
    Convert a six-character GRID utterance code into text.

    Example:
        pgag6a
        -> place green at g 6 again
    """

    code = code.lower().strip()

    if len(code) != 6:
        raise ValueError(
            f"Invalid GRID code: {code!r}. "
            "Expected exactly 6 characters."
        )

    command_code = code[0]
    color_code = code[1]
    preposition_code = code[2]
    letter = code[3]
    digit = code[4]
    adverb_code = code[5]

    try:
        command = COMMANDS[command_code]
        color = COLORS[color_code]
        preposition = PREPOSITIONS[preposition_code]
        adverb = ADVERBS[adverb_code]
    except KeyError as exc:
        raise ValueError(
            f"Unknown GRID code component in {code!r}: {exc}"
        ) from exc

    if not letter.isalpha():
        raise ValueError(
            f"Invalid letter component in GRID code: {code!r}"
        )

    if digit == "z":
        digit = "0"
    elif not digit.isdigit():
        raise ValueError(
            f"Invalid digit component in GRID code: {code!r}"
        )

    return (
        f"{command} "
        f"{color} "
        f"{preposition} "
        f"{letter} "
        f"{digit} "
        f"{adverb}"
    )


def decode_grid_file(video_path: Path | str) -> str:
    """
    Decode the GRID utterance code from a video filename.

    Example:
        /path/to/pgag6a.mpg
        -> place green at g 6 again
    """

    path = Path(video_path)

    return decode_grid_code(path.stem)