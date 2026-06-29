from preprocessing.frame_extractor import FrameExtractor
from preprocessing.face_detector import FaceDetector
from preprocessing.face_cropper import FaceCropper
from preprocessing.mouth_cropper import MouthCropper
from models.inference import LipReader

class ProcessingPipeline:

    def __init__(self, video_path):

        self.video_path = video_path

    def run(self):

        extractor = FrameExtractor(
            self.video_path,
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

        mouth = MouthCropper()

        mouths = mouth.crop_mouths(
            "outputs/cropped_faces",
            "outputs/mouths"
        )

        reader = LipReader()
        prediction = reader.predict(
             "outputs/mouths"
             )
        
        return {

    "frames": total_frames,
    "faces": processed,
    "cropped": cropped,
    "mouths": mouths,
    "prediction": prediction["text"],
    "confidence": prediction["confidence"]
}