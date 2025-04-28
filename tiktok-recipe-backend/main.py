import os
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import requests
import whisper
import pytesseract
from PIL import Image
import torch
from transformers import pipeline
from fpdf import FPDF
from fastapi.responses import FileResponse
import shutil
from pathlib import Path
from tiktok_scraper import TikTokScraper

# Create the FastAPI app
app = FastAPI(
    title="TikTok Recipe Extractor API",
    description="API for extracting recipe information from TikTok videos",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temporary directory for storing files
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# Load models
@app.on_event("startup")
async def startup_event():
    global whisper_model, nlp_pipeline
    # Load Whisper model for speech-to-text
    whisper_model = whisper.load_model("base")
    # Load NLP pipeline for text analysis
    nlp_pipeline = pipeline("text-classification", model="distilbert-base-uncased")

# Define request and response models
class TikTokRequest(BaseModel):
    url: str
    profil_utilisateur: Optional[str] = None

class Ingredient(BaseModel):
    nom: str
    quantité: str

class RecipeResponse(BaseModel):
    titre: str
    résumé: str
    ingrédients: List[Ingredient]
    étapes: List[str]
    temps: Dict[str, str]
    nutrition: Dict[str, float]
    évaluation_santé: str
    transcription: Optional[str] = None
    ocr_text: Optional[str] = None
    pdf_link: Optional[str] = None

# Helper functions
def download_tiktok_video(url: str) -> str:
    """Download TikTok video and return the path to the downloaded file."""
    try:
        from urllib.parse import urlparse

        # Create a unique filename based on the URL
        temp_file = TEMP_DIR / f"{hash(url)}.mp4"

        # Check if the URL is a TikTok URL or YouTube URL
        parsed_url = urlparse(url)

        if "youtube.com" in parsed_url.netloc or "youtu.be" in parsed_url.netloc:
            # If it's a YouTube URL, use pytube
            from pytube import YouTube

            # Download the video using pytube
            yt = YouTube(url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

            if not stream:
                raise ValueError("No suitable video stream found")

            # Download to the temporary file
            stream.download(output_path=str(TEMP_DIR), filename=f"{hash(url)}.mp4")
        elif "tiktok.com" in parsed_url.netloc:
            # For TikTok URLs, use the TikTokScraper library
            scraper = TikTokScraper()

            # Download the video
            video_path = scraper.download_video(url, str(TEMP_DIR))

            # Get video metadata
            try:
                video_metadata = scraper.get_video_metadata(url)
                # Log metadata for debugging
                import logging
                logging.info(f"TikTok video metadata: {video_metadata}")
            except Exception as metadata_error:
                # Log the error but continue with the download
                import logging
                logging.warning(f"Failed to get video metadata: {str(metadata_error)}")

            # If the download function returns a different path, copy it to our expected path
            if video_path != str(temp_file):
                # Get the filename from the returned path
                video_name = os.path.basename(video_path)

                # Rename the downloaded video with a prefix
                new_video_name = f"tiktok_{video_name}"
                new_video_path = os.path.join(os.path.dirname(video_path), new_video_name)

                try:
                    # Rename the file
                    os.rename(video_path, new_video_path)
                    video_path = new_video_path

                    # Log the rename operation
                    import logging
                    logging.info(f"Renamed video: {video_path}")
                except Exception as rename_error:
                    # Log the error but continue with the original path
                    import logging
                    logging.warning(f"Failed to rename video: {str(rename_error)}")

                # Copy the file to our expected path
                shutil.copy2(video_path, temp_file)

                # Optionally, remove the original file if it's in a different location
                if os.path.dirname(video_path) != str(TEMP_DIR):
                    os.remove(video_path)
        else:
            raise ValueError(f"Unsupported URL: {url}. Only TikTok and YouTube URLs are supported.")

        # Verify the file exists
        if not temp_file.exists():
            raise FileNotFoundError(f"Failed to download video to {temp_file}")

        return str(temp_file)
    except Exception as e:
        error_message = f"Failed to download video: {str(e)}"
        # Log the error for debugging
        import logging
        logging.error(f"Video download error: {error_message}")
        logging.error(f"URL: {url}")

        # Provide a more detailed error message
        if "403" in str(e):
            error_message = f"Access denied (403 Forbidden) when trying to download the video. This might be due to TikTok's anti-scraping measures. URL: {url}"
        elif "Could not find video URL" in str(e):
            error_message = f"Could not extract video URL from the TikTok page. The video might be private or the page structure might have changed. URL: {url}"
        elif "Failed to download video after" in str(e):
            error_message = f"Multiple download attempts failed. TikTok might be blocking automated access. URL: {url}"

        raise HTTPException(status_code=500, detail=error_message)

def extract_audio_from_video(video_path: str) -> str:
    """Extract audio from video and return the path to the audio file."""
    try:
        import subprocess

        # Create output audio path
        audio_path = video_path.replace(".mp4", ".wav")

        # Use ffmpeg to extract audio from video
        command = [
            "ffmpeg",
            "-i", video_path,  # Input file
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # Audio codec
            "-ar", "16000",  # Audio sample rate
            "-ac", "1",  # Mono audio
            "-y",  # Overwrite output file if it exists
            audio_path  # Output file
        ]

        # Run the ffmpeg command
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        # Check if the audio file was created
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Failed to extract audio to {audio_path}")

        return audio_path
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract audio with ffmpeg: {e.stderr.decode()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract audio: {str(e)}")

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using Whisper and return the transcription."""
    try:
        # Use the global Whisper model loaded at startup
        global whisper_model

        # Check if the audio file exists
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Transcribe the audio using Whisper
        result = whisper_model.transcribe(audio_path)

        # Extract the transcription text
        transcription = result["text"]

        if not transcription:
            raise ValueError("Transcription is empty")

        return transcription
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

def extract_text_from_frames(video_path: str) -> str:
    """Extract text from video frames using OCR and return the extracted text."""
    try:
        import subprocess
        import cv2
        import numpy as np
        from PIL import Image

        # Check if the video file exists
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create a directory for frames
        frames_dir = TEMP_DIR / f"frames_{hash(video_path)}"
        frames_dir.mkdir(exist_ok=True)

        # Extract frames from the video using ffmpeg
        # Extract 1 frame every 3 seconds
        extract_frames_command = [
            "ffmpeg",
            "-i", video_path,
            "-vf", "fps=1/3",  # 1 frame every 3 seconds
            "-q:v", "2",  # High quality
            str(frames_dir / "frame_%04d.jpg"),
            "-y"  # Overwrite existing files
        ]

        subprocess.run(
            extract_frames_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        # Get all extracted frames
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))

        if not frame_files:
            raise ValueError("No frames were extracted from the video")

        # Extract text from each frame using pytesseract
        all_text = []

        for frame_file in frame_files:
            # Open the image
            img = Image.open(frame_file)

            # Extract text using pytesseract
            text = pytesseract.image_to_string(img, lang='eng')

            if text.strip():
                all_text.append(text.strip())

        # Combine all extracted text
        combined_text = "\n".join(all_text)

        if not combined_text.strip():
            return "No text detected in video frames."

        return combined_text
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract frames with ffmpeg: {e.stderr.decode()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text from frames: {str(e)}")

def analyze_text(transcription: str, ocr_text: str, profil_utilisateur: Optional[str]) -> RecipeResponse:
    """Analyze text and return structured recipe information."""
    try:
        # Use the global NLP pipeline loaded at startup
        global nlp_pipeline

        # Combine transcription and OCR text for analysis
        combined_text = f"{transcription}\n\n{ocr_text}"

        # Use transformers to analyze the text
        # In a real-world scenario, you would fine-tune a model specifically for recipe extraction
        # Here we'll use a combination of rule-based extraction and NLP

        # Extract ingredients using pattern matching
        ingredients = []
        ingredient_patterns = [
            r'(\d+(?:\.\d+)?)\s*(g|kg|ml|l|cup|cups|tbsp|tsp|tablespoon|teaspoon|oz|ounce|pound|lb)?\s+(?:of\s+)?([a-zA-Z\s]+)',
            r'([a-zA-Z\s]+)(?:\s*:)?\s*(\d+(?:\.\d+)?)\s*(g|kg|ml|l|cup|cups|tbsp|tsp|tablespoon|teaspoon|oz|ounce|pound|lb)'
        ]

        import re
        for pattern in ingredient_patterns:
            matches = re.finditer(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 3:
                    quantity = f"{match.group(1)}{match.group(2) or ''}"
                    name = match.group(3).strip().lower()
                    ingredients.append(Ingredient(nom=name, quantité=quantity))
                elif len(match.groups()) >= 2:
                    name = match.group(1).strip().lower()
                    quantity = f"{match.group(2)}{match.group(3) or ''}"
                    ingredients.append(Ingredient(nom=name, quantité=quantity))

        # If no ingredients were found, add some default ones based on the text
        if not ingredients:
            # Look for common food items in the text
            common_foods = ["chicken", "poulet", "rice", "riz", "broccoli", "brocolis", "beef", "boeuf", 
                           "pasta", "pâtes", "tomato", "tomate", "onion", "oignon", "garlic", "ail"]

            for food in common_foods:
                if food in combined_text.lower():
                    ingredients.append(Ingredient(nom=food, quantité="as needed"))

        # Extract steps using sentence splitting
        steps = []
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', transcription)
        for sentence in sentences:
            # Filter for sentences that look like instructions
            if re.search(r'^(first|then|next|finally|after|before|now|start|begin|add|mix|stir|cook|bake|pour|cut)', 
                        sentence, re.IGNORECASE):
                steps.append(sentence.strip())

        # If no steps were found, extract sentences that might be instructions
        if not steps:
            for sentence in sentences:
                if len(sentence.split()) > 5 and any(word in sentence.lower() for word in 
                                                  ["add", "mix", "stir", "cook", "bake", "pour", "cut", "place", "heat"]):
                    steps.append(sentence.strip())

        # Extract preparation and cooking time
        prep_time = "15 min"  # Default
        cook_time = "25 min"  # Default

        prep_match = re.search(r'(?:preparation|prep)(?:\s+time)?(?:\s*:)?\s*(\d+)\s*(min|minute|minutes|hour|hours|h)', 
                              combined_text, re.IGNORECASE)
        if prep_match:
            prep_time = f"{prep_match.group(1)} {prep_match.group(2)}"

        cook_match = re.search(r'(?:cooking|cook)(?:\s+time)?(?:\s*:)?\s*(\d+)\s*(min|minute|minutes|hour|hours|h)', 
                              combined_text, re.IGNORECASE)
        if cook_match:
            cook_time = f"{cook_match.group(1)} {cook_match.group(2)}"

        # Generate a title based on the text
        title_candidates = []
        for sentence in sentences:
            if "recipe" in sentence.lower() or "how to make" in sentence.lower():
                title_candidates.append(sentence)

        title = "Recipe from TikTok"  # Default title
        if title_candidates:
            # Use the shortest candidate as the title
            title = min(title_candidates, key=len).strip()
            # Clean up the title
            title = re.sub(r'(recipe for|how to make|this is a recipe for)', '', title, flags=re.IGNORECASE).strip()
            title = title.rstrip('.!?')

        # Generate a summary
        summary = "A delicious recipe extracted from a TikTok video."
        if len(transcription) > 100:
            summary = transcription[:100].strip() + "..."

        # Estimate nutrition based on ingredients
        nutrition = {"calories": 0, "protéines": 0, "lipides": 0, "glucides": 0}

        # Very basic nutrition estimation
        for ingredient in ingredients:
            if "chicken" in ingredient.nom or "poulet" in ingredient.nom:
                nutrition["calories"] += 200
                nutrition["protéines"] += 25
                nutrition["lipides"] += 10
                nutrition["glucides"] += 0
            elif "rice" in ingredient.nom or "riz" in ingredient.nom:
                nutrition["calories"] += 150
                nutrition["protéines"] += 3
                nutrition["lipides"] += 0
                nutrition["glucides"] += 30
            elif "broccoli" in ingredient.nom or "brocolis" in ingredient.nom:
                nutrition["calories"] += 50
                nutrition["protéines"] += 3
                nutrition["lipides"] += 0
                nutrition["glucides"] += 10
            # Add more ingredients as needed

        # Health evaluation
        health_eval = "Recette équilibrée"
        if nutrition["protéines"] > 20:
            health_eval = "Recette riche en protéines"
        if nutrition["glucides"] > 50:
            health_eval += ", riche en glucides"
        if nutrition["lipides"] < 10:
            health_eval += ", faible en lipides"

        # Create the recipe response
        recipe = RecipeResponse(
            titre=title,
            résumé=summary,
            ingrédients=ingredients,
            étapes=steps if steps else ["Suivre les instructions de la vidéo"],
            temps={"préparation": prep_time, "cuisson": cook_time},
            nutrition=nutrition,
            évaluation_santé=health_eval,
            transcription=transcription,
            ocr_text=ocr_text
        )

        # Customize evaluation based on user profile
        if profil_utilisateur:
            if profil_utilisateur.lower() == "perte de poids":
                if nutrition["calories"] < 500:
                    recipe.évaluation_santé = "Adapté à la perte de poids : faible en calories, " + health_eval.lower()
                else:
                    recipe.évaluation_santé = "Modérément adapté à la perte de poids : " + health_eval.lower()
            elif profil_utilisateur.lower() == "prise de masse":
                if nutrition["protéines"] > 30 and nutrition["calories"] > 500:
                    recipe.évaluation_santé = "Très adapté à la prise de masse : riche en protéines et calories"
                else:
                    recipe.évaluation_santé = "Modérément adapté à la prise de masse : " + health_eval.lower()
            elif profil_utilisateur.lower() == "végétarien":
                if any("chicken" in i.nom or "poulet" in i.nom or "beef" in i.nom or "boeuf" in i.nom 
                      for i in ingredients):
                    recipe.évaluation_santé = "Non adapté aux végétariens : contient de la viande"
                else:
                    recipe.évaluation_santé = "Adapté aux végétariens : " + health_eval.lower()

        return recipe
    except Exception as e:
        # If analysis fails, return a basic recipe with the transcription and OCR text
        basic_recipe = RecipeResponse(
            titre="Recette extraite de TikTok",
            résumé="Recette extraite automatiquement d'une vidéo TikTok.",
            ingrédients=[Ingredient(nom="Ingrédients", quantité="voir transcription")],
            étapes=["Voir la transcription pour les étapes détaillées"],
            temps={"préparation": "N/A", "cuisson": "N/A"},
            nutrition={"calories": 0, "protéines": 0, "lipides": 0, "glucides": 0},
            évaluation_santé="Information non disponible",
            transcription=transcription,
            ocr_text=ocr_text
        )
        return basic_recipe

def generate_pdf(recipe: RecipeResponse) -> str:
    """Generate a PDF with the recipe information and return the path to the PDF."""
    try:
        pdf = FPDF()
        pdf.add_page()

        # Set font
        pdf.set_font("Arial", "B", 16)

        # Title
        pdf.cell(0, 10, recipe.titre, 0, 1, "C")

        # Summary
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, recipe.résumé)

        # Ingredients
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Ingrédients:", 0, 1)
        pdf.set_font("Arial", "", 12)
        for ingredient in recipe.ingrédients:
            pdf.cell(0, 10, f"- {ingredient.nom}: {ingredient.quantité}", 0, 1)

        # Steps
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Étapes:", 0, 1)
        pdf.set_font("Arial", "", 12)
        for i, step in enumerate(recipe.étapes, 1):
            pdf.multi_cell(0, 10, f"{i}. {step}")

        # Time
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Temps:", 0, 1)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Préparation: {recipe.temps['préparation']}", 0, 1)
        pdf.cell(0, 10, f"Cuisson: {recipe.temps['cuisson']}", 0, 1)

        # Nutrition
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Nutrition:", 0, 1)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Calories: {recipe.nutrition['calories']} kcal", 0, 1)
        pdf.cell(0, 10, f"Protéines: {recipe.nutrition['protéines']} g", 0, 1)
        pdf.cell(0, 10, f"Lipides: {recipe.nutrition['lipides']} g", 0, 1)
        pdf.cell(0, 10, f"Glucides: {recipe.nutrition['glucides']} g", 0, 1)

        # Health evaluation
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Évaluation santé:", 0, 1)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, recipe.évaluation_santé)

        # Save PDF
        pdf_path = TEMP_DIR / f"{hash(recipe.titre)}.pdf"
        pdf.output(str(pdf_path))

        return str(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

# Define API endpoints
@app.post("/extract", response_model=RecipeResponse)
async def extract_recipe(request: TikTokRequest):
    """Extract recipe information from a TikTok video."""
    try:
        # Download video
        video_path = download_tiktok_video(request.url)

        # Extract audio
        audio_path = extract_audio_from_video(video_path)

        # Transcribe audio
        transcription = transcribe_audio(audio_path)

        # Extract text from frames
        ocr_text = extract_text_from_frames(video_path)

        # Analyze text
        recipe = analyze_text(transcription, ocr_text, request.profil_utilisateur)

        # Generate PDF
        pdf_path = generate_pdf(recipe)

        # Set PDF link
        recipe.pdf_link = f"/download/{Path(pdf_path).name}"

        return recipe
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_pdf(filename: str):
    """Download a PDF file."""
    file_path = TEMP_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), media_type="application/pdf", filename="recipe.pdf")

@app.post("/download-video")
async def download_video(request: TikTokRequest):
    """Download a TikTok video and return the video file."""
    try:
        # Download video using the existing function
        video_path = download_tiktok_video(request.url)

        # Create a unique filename for the downloaded video
        video_filename = f"tiktok_video_{hash(request.url)}.mp4"

        # Copy the video to a new location with a more user-friendly name
        user_video_path = TEMP_DIR / video_filename
        shutil.copy2(video_path, user_video_path)

        # Return the path to the video file for frontend to access
        return {"video_link": f"/download-video-file/{video_filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-video-file/{filename}")
async def download_video_file(filename: str):
    """Download a video file."""
    file_path = TEMP_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(str(file_path), media_type="video/mp4", filename=filename)

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the TikTok Recipe Extractor API"}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
