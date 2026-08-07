from pathlib import Path
import shutil

from fastapi import UploadFile

from app.config import ALLOWED_EXTENSIONS, UPLOAD_DIR


def is_allowed_file(filename: str) -> bool:
    """
    Check whether the uploaded file has a supported video extension.
    """

    extension = Path(filename).suffix.lower()

    return extension in ALLOWED_EXTENSIONS


def save_uploaded_file(video: UploadFile) -> Path:
    """
    Save the uploaded video to the uploads directory.
    Returns the saved file path.
    """

    destination = UPLOAD_DIR / video.filename

    with open(destination, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    return destination


def get_file_size(file_path: Path) -> float:
    """
    Return file size in MB.
    """

    size = file_path.stat().st_size

    return round(size / (1024 * 1024), 2)