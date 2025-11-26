# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from whisper_api import speechToText
from elevenlabs_api import textToSpeech
from openai_api import askGPT, evaluateCoherence
import os

app = Flask(__name__)
CORS(app)  # Pour autoriser les appels depuis React

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
    
    return jsonify({
        "response": response,
        "coherence": coherence_score
    })


if __name__ == "__main__":
    app.run(port=5000)
