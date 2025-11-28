import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# Configuration Hugging Face API
HF_TOKEN = os.getenv('HF_TOKEN')

# Modèles utilisés
CHAT_MODEL = "Qwen/Qwen2.5-72B-Instruct"  # Modèle de chat (puissant et gratuit via API)
SIMILARITY_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Pour la similarité

# Créer le client Hugging Face
client = InferenceClient(token=HF_TOKEN)

print(f"Configuration Hugging Face API")
print(f"  - Modèle de chat: {CHAT_MODEL}")
print(f"  - Modèle de similarité: {SIMILARITY_MODEL}")


def askGPT(messages: list):
    """Génère une réponse avec l'API Hugging Face basée sur l'historique des messages."""
    try:
        response = client.chat_completion(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )
        
        answer = response.choices[0].message.content.strip()
        
        print("RÉPONSE DU LLM :\n", answer)
        return answer
    except Exception as e:
        print(f"Erreur lors de la génération: {e}")
        raise e


def evaluateCoherence(question: str, answer: str) -> int:
    """
    Évalue la cohérence entre une question et une réponse (0-100%).
    Utilise l'API Hugging Face pour la similarité sémantique.
    """
    try:
        # Utiliser l'API sentence_similarity de Hugging Face
        similarities = client.sentence_similarity(
            sentence=question,
            other_sentences=[answer],
            model=SIMILARITY_MODEL
        )
        
        # Le résultat est une liste avec un score pour chaque phrase comparée
        similarity = similarities[0]
        
        # La similarité va de 0 à 1, on la convertit en pourcentage 0-100
        # avec une transformation pour avoir des scores plus significatifs
        if similarity < 0:
            score = 0
        elif similarity < 0.15:
            score = int(similarity * 100)  # 0-15%
        elif similarity < 0.3:
            score = int(15 + (similarity - 0.15) * 200)  # 15-45%
        elif similarity < 0.5:
            score = int(45 + (similarity - 0.3) * 200)  # 45-85%
        else:
            score = int(85 + (similarity - 0.5) * 30)  # 85-100%
        
        score = max(0, min(100, score))  # Clamp entre 0 et 100
        
        print(f"SIMILARITÉ SÉMANTIQUE: {similarity:.3f}")
        print(f"SCORE DE COHÉRENCE: {score}%")
        return score
    except Exception as e:
        print(f"Erreur lors de l'évaluation de cohérence: {e}")
        return 50  # Valeur par défaut en cas d'erreur
