<div align="center">

# LipVision

### AI-Powered Lip Reading & Visual Speech Recognition System

## Developed by

**Kushal Sarkar**
🎓 Diploma in Computer Science & Engineering
GitHub: https://github.com/ByteBender9
</div>

Transform silent lip movements into readable text using Computer Vision and Artificial Intelligence.

🚧 **Currently Under Active Development**

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=for-the-badge&logo=opencv)
![Status](https://img.shields.io/badge/Status-Development-orange?style=for-the-badge)

</div>

---

# 📖 Overview

LipVision is an AI-powered visual speech recognition project designed to convert lip movements from video into readable text.

The application processes uploaded videos through a modular computer vision pipeline, preparing the data for deep learning-based lip reading.

The long-term goal is to build a complete end-to-end Visual Speech Recognition (VSR) system capable of generating subtitles from silent videos.

---

# ✨ Current Features

### Video Processing

- Upload videos through a FastAPI web interface
- Automatic video metadata extraction
- Frame extraction
- Face detection
- Face cropping
- Mouth region extraction

### Backend

- FastAPI backend
- Modular processing pipeline
- Organized project architecture

### Development

- Git version control
- GitHub integration
- Clean project structure

---

# 🚀 Planned Features

- Pretrained AI Lip Reading Model
- Subtitle (.srt) Generation
- Video Export with Subtitles
- Confidence Score
- Progress Bar
- Processing History
- Modern Dashboard
- Drag & Drop Upload
- Real-time Processing

---

# 🏗 Project Architecture

```
                Video Upload
                      │
                      ▼
            Video Metadata Extraction
                      │
                      ▼
              Frame Extraction
                      │
                      ▼
              Face Detection
                      │
                      ▼
               Face Cropping
                      │
                      ▼
             Mouth Region Crop
                      │
                      ▼
          AI Lip Reading Model
                      │
                      ▼
             Subtitle Generation
                      │
                      ▼
              Download Results
```

---

# 📂 Project Structure

```
LipVision
│
├── app/
│   ├── static/
│   └── templates/
│
├── preprocessing/
│   ├── frame_extractor.py
│   ├── face_detector.py
│   ├── face_cropper.py
│   └── mouth_cropper.py
│
├── services/
│   └── pipeline.py
│
├── models/
│   ├── pretrained/
│   └── inference.py
│
├── uploads/
├── outputs/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/ByteBender9/LipVision.git
```

Go inside the project

```bash
cd LipVision
```

Create Virtual Environment

```bash
python -m venv .venv
```

Activate Environment

### macOS/Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
uvicorn app:app --reload
```

Open

```
http://127.0.0.1:8000
```

---

# 🛠 Technology Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Backend Framework | FastAPI |
| Computer Vision | OpenCV |
| Version Control | Git, GitHub |
| Development Environment | VS Code |

---

# 📌 Development Roadmap

## Phase 1 — Foundation ✅

- [x] FastAPI Setup
- [x] Video Upload
- [x] Metadata Extraction
- [x] Frame Extraction
- [x] Face Detection
- [x] Face Cropping
- [x] Mouth Cropping

---

## Phase 2 — Application 🚧

- [ ] Processing Dashboard
- [ ] Better UI
- [ ] Progress Indicator
- [ ] Results Page

---

## Phase 3 — AI Integration

- [ ] Pretrained Lip Reading Model
- [ ] Visual Speech Recognition
- [ ] Subtitle Generation

---

## Phase 4 — Production

- [ ] Video Export
- [ ] Download Results
- [ ] Performance Optimization
- [ ] Deployment

---

## 📸 Screenshots

### Home Page
Coming Soon

### Processing Pipeline
Coming Soon

### Results Page
Coming Soon

---

# 📊 Current Progress

| Module | Status |
|----------|--------|
| Upload | ✅ |
| Metadata | ✅ |
| Frame Extraction | ✅ |
| Face Detection | ✅ |
| Face Cropping | ✅ |
| Mouth Cropping | ✅ |
| AI Model | 🚧 |
| Subtitle Generation | ⏳ |
| Deployment | ⏳ |

---

# 🤝 Contributing

Contributions, ideas, and suggestions are welcome.

If you'd like to improve LipVision:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

## 🚧 Project Status

LipVision is currently under active development.

Features, project structure, and APIs may change as development progresses.

---

## 📬 Contact

**Kushal Sarkar**
- GitHub: https://github.com/ByteBender9
- LinkedIn: https://www.linkedin.com/in/kushalsarkar
- Email: connect.kushals@gmail.com

---

# ⭐ Support

If you found this project useful or interesting, consider giving it a ⭐ on GitHub.

It helps support the project and motivates future development.

---

<div align="center">

**Built with ❤️ using Python, FastAPI, OpenCV and AI**

</div>