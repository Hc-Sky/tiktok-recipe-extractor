import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the TikTok Recipe Extractor API"}

def test_extract_recipe():
    """Test the extract recipe endpoint."""
    response = client.post(
        "/extract",
        json={"url": "https://www.tiktok.com/@easycooking/video/123456789"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "titre" in data
    assert "résumé" in data
    assert "ingrédients" in data
    assert "étapes" in data
    assert "temps" in data
    assert "nutrition" in data
    assert "évaluation_santé" in data
    assert "transcription" in data
    assert "ocr_text" in data
    assert "pdf_link" in data

def test_extract_recipe_with_profile():
    """Test the extract recipe endpoint with a user profile."""
    response = client.post(
        "/extract",
        json={"url": "https://www.tiktok.com/@easycooking/video/123456789", "profil_utilisateur": "prise de masse"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["évaluation_santé"] == "Très adapté à la prise de masse : riche en protéines et glucides complexes"

def test_download_pdf_not_found():
    """Test the download PDF endpoint with a non-existent file."""
    response = client.get("/download/nonexistent.pdf")
    assert response.status_code == 404
    assert response.json() == {"detail": "File not found"}