# TikTok Recipe Extractor Backend

This is the backend for the TikTok Recipe Extractor application. It's built with FastAPI and provides endpoints for extracting recipe information from TikTok videos.

## Features

- Extract recipe information from TikTok videos
- Transcribe audio using Whisper
- Extract text from video frames using OCR
- Generate structured recipe data with ingredients, steps, nutrition info, etc.
- Evaluate recipes based on user profiles (e.g., weight loss, muscle gain, vegetarian)

## Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Usage

1. Start the server: `uvicorn main:app --reload`
2. The API will be available at `http://localhost:8000`
3. Access the API documentation at `http://localhost:8000/docs`

## API Endpoints

- `POST /extract`: Extract recipe information from a TikTok video
  - Request body: `{"url": "https://www.tiktok.com/@user/video/123456789", "profil_utilisateur": "prise de masse"}`
  - Response: JSON with recipe information

## Dependencies

- FastAPI
- Pydantic
- TikTokAPI
- Whisper
- Pytesseract
- Transformers
- FPDF