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
        import re
        import requests
        import time
        from urllib.parse import urlparse

        # Create a unique filename based on the URL
        temp_file = TEMP_DIR / f"{hash(url)}.mp4"

        # Check if the URL is a TikTok URL or TikTok CDN URL
        parsed_url = urlparse(url)
        is_tiktok_url = any(domain in parsed_url.netloc for domain in ["tiktok.com", "tiktok", "v16-webapp", "v16.tiktokcdn"])
        is_direct_cdn_url = any(domain in parsed_url.netloc for domain in ["v16-webapp", "v16.tiktokcdn"]) and parsed_url.path.endswith(('.mp4', '.mov'))

        if not is_tiktok_url:
            # If not a TikTok URL, check if it's a YouTube URL
            if "youtube.com" in parsed_url.netloc or "youtu.be" in parsed_url.netloc:
                from pytube import YouTube

                # Download the video using pytube
                yt = YouTube(url)
                stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

                if not stream:
                    raise ValueError("No suitable video stream found")

                # Download to the temporary file
                stream.download(output_path=str(TEMP_DIR), filename=f"{hash(url)}.mp4")
            else:
                raise ValueError(f"Unsupported URL: {url}. Only TikTok and YouTube URLs are supported.")
        else:
            # For TikTok URLs, we'll use a session-based approach with more realistic browser headers
            session = requests.Session()

            # Special handling for direct CDN URLs
            if is_direct_cdn_url:
                # For direct CDN URLs, we'll use specialized headers
                cdn_headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "video/webm,video/mp4,video/*;q=0.9,application/ogg;q=0.7,audio/*;q=0.6,*/*;q=0.5",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Referer": "https://www.tiktok.com/",
                    "Origin": "https://www.tiktok.com",
                    "Sec-Fetch-Dest": "video",
                    "Sec-Fetch-Mode": "no-cors",
                    "Sec-Fetch-Site": "cross-site",
                    "Range": "bytes=0-",
                    "Pragma": "no-cache",
                    "Cache-Control": "no-cache"
                }

                # Try multiple user agents if one fails
                user_agents = [
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
                ]

                # Try each user agent
                for user_agent in user_agents:
                    try:
                        cdn_headers["User-Agent"] = user_agent

                        # Download the video with streaming
                        response = session.get(
                            url, 
                            headers=cdn_headers,
                            timeout=15,
                            stream=True
                        )

                        if response.status_code == 200:
                            # Save the video to a file
                            with open(temp_file, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)

                            # If successful, break the loop
                            if temp_file.exists() and temp_file.stat().st_size > 10240:
                                break

                        # If we got a 403, try the next user agent
                        if response.status_code == 403:
                            continue

                        # For other errors, try a different approach
                        if response.status_code != 200:
                            break

                    except requests.exceptions.RequestException:
                        # If there's an error, try the next user agent
                        continue

                # If we couldn't download the video directly, try using yt-dlp
                if not temp_file.exists() or temp_file.stat().st_size < 10240:
                    try:
                        import subprocess
                        # Try using yt-dlp as a fallback
                        subprocess.run(
                            ["yt-dlp", "-o", str(temp_file), url],
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                    except (subprocess.SubprocessError, FileNotFoundError):
                        # If yt-dlp fails or is not installed, raise an error
                        raise ValueError(f"Failed to download video from CDN URL: {url}")

                # If we've successfully downloaded the video, return the path
                if temp_file.exists() and temp_file.stat().st_size > 0:
                    return str(temp_file)

            # Set up headers that mimic a real browser more closely
            browser_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
                "Referer": "https://www.google.com/"
            }

            # First request to get the TikTok page
            response = session.get(url, headers=browser_headers, timeout=10)
            if response.status_code != 200:
                raise ValueError(f"Failed to access TikTok URL: {url}, status code: {response.status_code}")

            # Small delay to mimic human behavior
            time.sleep(1)

            # Extract the video URL from the HTML
            video_url_match = re.search(r'"playAddr":"([^"]+)"', response.text)
            if not video_url_match:
                video_url_match = re.search(r'"downloadAddr":"([^"]+)"', response.text)

            if not video_url_match:
                # Try alternative patterns
                video_url_match = re.search(r'<video[^>]*src="([^"]+)"', response.text)

            if not video_url_match:
                raise ValueError(f"Could not find video URL in TikTok page: {url}")

            video_url = video_url_match.group(1).replace('\\u002F', '/').replace('\\/', '/')

            # Update headers for video request
            video_headers = browser_headers.copy()
            video_headers.update({
                "Referer": url,  # Set the TikTok URL as the referer
                "Range": "bytes=0-",  # Request the whole file
                "Sec-Fetch-Dest": "video",
                "Sec-Fetch-Mode": "no-cors"
            })

            # Try to download with multiple attempts
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Download the video
                    video_response = session.get(
                        video_url, 
                        headers=video_headers, 
                        timeout=15,
                        stream=True  # Stream the response to handle large files
                    )

                    # Check if successful
                    if video_response.status_code == 200:
                        # Save the video to a file
                        with open(temp_file, 'wb') as f:
                            for chunk in video_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        break
                    elif video_response.status_code == 403 and attempt < max_attempts - 1:
                        # If forbidden and not the last attempt, try with a different user agent
                        time.sleep(2)  # Wait before retry
                        video_headers["User-Agent"] = [
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
                            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        ][attempt]
                        continue
                    else:
                        raise ValueError(f"Failed to download video from URL: {video_url}, status code: {video_response.status_code}")
                except requests.exceptions.RequestException as e:
                    if attempt < max_attempts - 1:
                        time.sleep(2)  # Wait before retry
                        continue
                    raise ValueError(f"Network error while downloading video: {str(e)}")

            # If we've tried all attempts and still failed
            if not temp_file.exists():
                raise ValueError(f"Failed to download video after {max_attempts} attempts")

            # If the file is too small (less than 10KB), it might be an error page
            if temp_file.stat().st_size < 10240:
                with open(temp_file, 'r', errors='ignore') as f:
                    content = f.read(1000)
                    if '<html' in content.lower() or '<!doctype html' in content.lower():
                        raise ValueError(f"Downloaded content appears to be HTML, not a video file")

            # Verify it's a valid video file
            if temp_file.exists() and temp_file.stat().st_size > 0:
                # Basic check to see if it's a video file (check for MP4 signature)
                with open(temp_file, 'rb') as f:
                    header = f.read(12)
                    if not (header[4:8] == b'ftyp' and (header[8:12] == b'mp42' or header[8:12] == b'isom')):
                        # Not a valid MP4 file, try direct download as a fallback
                        try:
                            import subprocess
                            # Try using youtube-dl or yt-dlp as a fallback
                            subprocess.run(
                                ["yt-dlp", "-o", str(temp_file), url],
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                        except (subprocess.SubprocessError, FileNotFoundError):
                            # If yt-dlp fails or is not installed, raise the original error
                            raise ValueError(f"Downloaded file is not a valid video file")
            else:
                raise FileNotFoundError(f"Failed to download video to {temp_file}")

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

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the TikTok Recipe Extractor API"}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
