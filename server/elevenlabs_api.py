from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from dotenv import load_dotenv
import os

load_dotenv()

client = ElevenLabs(
  api_key=os.getenv('ELEVENLABS_API_KEY')
)

def textToSpeech(text: str):
    try:
        # Utiliser la nouvelle API text_to_speech.convert
        audio = client.text_to_speech.convert(
            text=text,
            voice_id="MNiuKciqE420DCRJtdeb",  # voice_id au lieu de voice
            model_id="eleven_multilingual_v2",  # model_id au lieu de model
            output_format="mp3_44100_128"  # Spécifier le format de sortie
        )
        
        play(audio)
        return audio

    except Exception as e:
        print(f"Erreur lors de la synthèse vocale: {e}")
        raise e