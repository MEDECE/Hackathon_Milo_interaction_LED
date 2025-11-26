import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# Configuration Hugging Face API
HF_TOKEN = os.getenv('HF_TOKEN')

# Modèle Whisper pour la transcription
WHISPER_MODEL = "openai/whisper-large-v3"

# Créer le client Hugging Face
client = InferenceClient(
    token=HF_TOKEN,
)

print(f"Configuration Whisper API: {WHISPER_MODEL}")


def speechToText(file):
    """Transcrit un fichier audio en texte via l'API Hugging Face."""
    try:
        # Lire le fichier audio
        with open(file, "rb") as audio_file:
            audio_data = audio_file.read()
        
        # Appeler l'API de transcription
        result = client.automatic_speech_recognition(
            audio=audio_data,
            model=WHISPER_MODEL
        )
        
        # Le résultat peut être un dict ou une string selon la version
        if isinstance(result, dict):
            text = result.get("text", "")
        else:
            text = str(result)
        
        print("QUESTION POSÉE :\n", text)
        return text
    except Exception as e:
        print(f"Erreur lors de la transcription: {e}")
        raise e
