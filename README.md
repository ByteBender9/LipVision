LipVision

AI-powered lip reading application that converts visual speech from videos into text using computer vision and deep learning.

⸻

📖 Overview

LipVision is a computer vision and AI project that aims to recognize spoken words by analyzing lip movements from video input. The application processes uploaded videos, extracts facial information, isolates the mouth region, and is being developed to generate text using a pretrained lip-reading model.

This project is built as a portfolio project to explore Computer Vision, Machine Learning, and AI-powered Visual Speech Recognition.

⸻

✨ Current Features

* 🎥 Video Upload
* 📹 Video Metadata Extraction
* 🖼️ Frame Extraction
* 😀 Face Detection
* ✂️ Face Cropping
* 👄 Mouth Region Cropping
* ⚙️ Modular Processing Pipeline
* 🌐 FastAPI Web Interface

⸻

🚀 Upcoming Features

* 🤖 Pretrained AI Lip Reading Model
* 📝 Subtitle (.srt) Generation
* 🎬 Export Video with Subtitles
* 📊 Processing Progress Bar
* 📁 Download Processed Results
* 📈 Improved User Interface

⸻

🛠️ Tech Stack

* Python
* FastAPI
* OpenCV
* TensorFlow (planned for AI inference)
* Computer Vision
* Machine Learning

⸻

📂 Project Structure

LipVision/
│
├── app/
├── models/
├── preprocessing/
├── services/
├── uploads/
├── outputs/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

⸻

⚙️ Installation

git clone https://github.com/ByteBender9/LipVision.git
cd LipVision
python -m venv .venv
source .venv/bin/activate ---------> # macOS/Linux
pip install -r requirements.txt
uvicorn app:app --reload

Open your browser and visit:

http://127.0.0.1:8000

⸻

🗺️ Development Roadmap

* Video Upload
* Video Metadata Extraction
* Frame Extraction
* Face Detection
* Face Cropping
* Mouth Cropping
* AI Lip Reading
* Subtitle Generation
* Export Processed Video

⸻

📌 Project Status

🚧 This project is currently under active development. New features, AI integration, and UI improvements are being added regularly.

⸻

👨‍💻 Author

Kushal Sarkar

GitHub: https://github.com/ByteBender9

⸻

⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!