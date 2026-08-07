from pathlib import Path

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"

OUTPUT_DIR = BASE_DIR / "outputs"

FRAME_DIR = OUTPUT_DIR / "frames"

FACE_DIR = OUTPUT_DIR / "faces"

CROPPED_FACE_DIR = OUTPUT_DIR / "cropped_faces"

MOUTH_DIR = OUTPUT_DIR / "mouths"


# --------------------------------------------------
# Allowed Video Formats
# --------------------------------------------------

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}


# --------------------------------------------------
# Directory Initialization
# --------------------------------------------------

DIRECTORIES = [
    UPLOAD_DIR,
    OUTPUT_DIR,
    FRAME_DIR,
    FACE_DIR,
    CROPPED_FACE_DIR,
    MOUTH_DIR
]


def create_directories():
    """
    Create all required project directories if they
    don't already exist.
    """

    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)