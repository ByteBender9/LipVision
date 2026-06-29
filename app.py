from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from preprocessing.frame_extractor import FrameExtractor
from preprocessing.face_detector import FaceDetector
from preprocessing.face_cropper import FaceCropper
from fastapi import FastAPI, Request, UploadFile, File, Form
from services.pipeline import ProcessingPipeline

import cv2
import os
import shutil

app = FastAPI(title="LipVision AI")

templates = Jinja2Templates(directory="app/templates")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/upload")
async def upload_video(request: Request, video: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, video.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    cap = cv2.VideoCapture(file_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = round(frames / fps, 2) if fps > 0 else 0

    cap.release()

    extractor = FrameExtractor(
        file_path,
        "outputs/frames"
    )

    total_frames = extractor.extract()

    detector = FaceDetector()

    processed = detector.detect_faces(
        "outputs/frames",
        "outputs/faces"
    )

    cropper = FaceCropper()

    cropped = cropper.crop_faces(
        "outputs/frames",
        "outputs/cropped_faces"
    )
    print("Cropped Faces:", cropped)
    print("Finished cropping")

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "filename": video.filename,
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "duration": duration,
            "frames": total_frames,
            "processed": processed,
            "cropped": cropped
        }
    )

@app.post("/process")
async def process_video(filename: str = Form(...)):

    video_path = f"uploads/{filename}"

    pipeline = ProcessingPipeline(video_path)

    result = pipeline.run()

    return result