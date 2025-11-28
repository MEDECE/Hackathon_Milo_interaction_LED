# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from whisper_api import speechToText
from elevenlabs_api import textToSpeech
from openai_api import askGPT, evaluateCoherence
from arduino_controller import init_arduino, send_coherence_score
import os

app = Flask(__name__)
CORS(app)  # Pour autoriser les appels depuis React

# Init Arduino (auto-detect by default or set ARDUINO_PORT in .env)
ARDUINO_PORT = os.getenv("ARDUINO_PORT", None)
init_arduino(ARDUINO_PORT)
@app.route("/api/speech-to-text", methods=["POST"])
def handle_speech_to_text():
    audio_file = request.files["file"]

    # Sauvegarder temporairement le fichier pour Whisper
    audio_path = "temp_audio.wav"
    audio_file.save(audio_path)

    result_text = speechToText(audio_path)

    os.remove(audio_path)

    return jsonify({"text": result_text})

@app.route("/api/text-to-speech", methods=["POST"])
def handle_text_to_speech():
    text = request.json["text"]
    textToSpeech(text)
    return jsonify({"success": True})

@app.route("/api/ask-gpt", methods=["POST"])
def handle_ask_gpt():
    messages = request.json["messages"]
    response = askGPT(messages)
    
    # Extraire la dernière question de l'utilisateur pour évaluer la cohérence
    last_user_message = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            last_user_message = msg["content"]
            break
    
    # Calculer le score de cohérence
    coherence_score = evaluateCoherence(last_user_message, response)

    # Envoyer le score de cohérence à l'Arduino (si connecté / sinon mode simulation)
    try:
        send_coherence_score(coherence_score)
    except Exception as e:
        print(f"Erreur lors de l'envoi du score à l'Arduino: {e}")

    return jsonify({
        "response": response,
        "coherence": coherence_score
    })


if __name__ == "__main__":
    app.run(port=5000)
