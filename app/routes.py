from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from services.processing_pipeline import ProcessingPipeline
from utils.file_utils import (
    get_file_size,
    is_allowed_file,
    save_uploaded_file,
)

router = APIRouter()

templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@router.post("/upload", response_class=HTMLResponse)
async def upload_video(
    request: Request,
    video: UploadFile = File(...)
):

    if not is_allowed_file(video.filename):

        raise HTTPException(
            status_code=400,
            detail="Unsupported video format.",
        )

    video_path = save_uploaded_file(video)

    file_size = get_file_size(video_path)

    pipeline = ProcessingPipeline(video_path)

    result = pipeline.run()

    metadata = result["metadata"]

    frame_result = result["frames"]

    detection_result = result["detections"]

    face_result = result["faces"]

    mouth_result = result["mouths"]

    return templates.TemplateResponse(
        request=request,
        name="processing.html",
        context={
            "filename": video.filename,
            "file_size": file_size,

            **metadata,

            "extracted_frames": frame_result["frame_count"],

            "detected_faces": detection_result.count,

            "cropped_faces": face_result["count"],

            "cropped_mouths": mouth_result["count"],
        },
    )