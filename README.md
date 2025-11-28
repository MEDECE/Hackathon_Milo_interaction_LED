# 🎤 Assistant Vocal ECE Paris

Assistant vocal intelligent pour l'ECE Paris, utilisant l'IA pour répondre aux questions des étudiants avec évaluation de la cohérence des réponses et feedback visuel par LEDs.

## 📋 Table des matières

- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Lancement](#-lancement)
- [Configuration Arduino (optionnel)](#-configuration-arduino-optionnel)
- [Utilisation](#-utilisation)
- [Dépannage](#-dépannage)

---

## ✨ Fonctionnalités

- 🎙️ **Reconnaissance vocale** : Transcription audio via Whisper (API Hugging Face)
- 🤖 **Génération de réponses** : Réponses intelligentes via Qwen 2.5 72B (API Hugging Face)
- 📊 **Score de cohérence** : Évaluation automatique de la pertinence des réponses (0-100%)
- 🔊 **Synthèse vocale** : Réponses parlées via ElevenLabs
- 💡 **Feedback LED** : Indicateur visuel de cohérence (vert/jaune/rouge) via Arduino

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │     │    Backend      │     │   APIs Cloud    │
│   (React)       │◄───►│    (Flask)      │◄───►│                 │
│   Port 3000     │     │    Port 5000    │     │  - Hugging Face │
└─────────────────┘     └────────┬────────┘     │  - ElevenLabs   │
                                 │              └─────────────────┘
                                 ▼
                        ┌─────────────────┐
                        │    Arduino      │
                        │    (LEDs)       │
                        └─────────────────┘
```

---

## 📦 Prérequis

### Logiciels à installer

| Logiciel | Version | Lien de téléchargement |
|----------|---------|------------------------|
| **Node.js** | 18+ | https://nodejs.org/ |
| **Python** | 3.10+ | https://www.python.org/downloads/ |
| **Git** | Dernière | https://git-scm.com/downloads |
| **Arduino IDE** | 2.x (optionnel) | https://www.arduino.cc/en/software |

### Comptes requis (gratuits)

| Service | Usage | Inscription |
|---------|-------|-------------|
| **Hugging Face** | LLM + Whisper + Similarité | https://huggingface.co/join |
| **ElevenLabs** | Synthèse vocale | https://elevenlabs.io/ |

---

## 🛠️ Installation

### 1. Cloner le projet

```bash
git clone https://github.com/MEDECE/Hackathon_Milo_interaction_LED.git
cd Hackathon_Milo_interaction_LED
```

### 2. Installer les dépendances Frontend (React)

```bash
npm install
```

### 3. Installer les dépendances Backend (Python)

```bash
cd server
pip install -r requirements.txt
cd ..
```

> **Note Windows** : Si `pip` ne fonctionne pas, essayez `python -m pip install -r requirements.txt`

---

## ⚙️ Configuration

### 1. Créer le fichier d'environnement

```bash
cd server
copy .env.example .env
```

> **Linux/Mac** : `cp .env.example .env`

### 2. Obtenir les clés API

#### 🤗 Token Hugging Face (gratuit)

1. Aller sur https://huggingface.co/settings/tokens
2. Se connecter ou créer un compte
3. Cliquer sur **"Create new token"**
4. Nom : `assistant-vocal` (ou autre)
5. Type : **Read**
6. Copier le token (commence par `hf_...`)

#### 🔊 Clé ElevenLabs (gratuit avec limite)

1. Aller sur https://elevenlabs.io/
2. Se connecter ou créer un compte
3. Aller dans **Profile Settings** → **API Keys**
4. Copier la clé API

### 3. Remplir le fichier `.env`

Ouvrir `server/.env` et remplacer les valeurs :

```env
# Token Hugging Face
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Clé ElevenLabs
ELEVENLABS_API_KEY=sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 🚀 Lancement

### Terminal 1 : Démarrer le serveur Backend

```bash
cd server
python app.py
```

Vous devriez voir :
```
Configuration Whisper API: openai/whisper-large-v3
Configuration Hugging Face API
  - Modèle de chat: Qwen/Qwen2.5-72B-Instruct
  - Modèle de similarité: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
 * Running on http://127.0.0.1:5000
```

### Terminal 2 : Démarrer le Frontend

```bash
npm start
```

L'application s'ouvre automatiquement sur http://localhost:3000

---

## 💡 Configuration Arduino (optionnel)

Cette section permet d'ajouter un feedback visuel avec des LEDs.

### Matériel nécessaire

- 1x Arduino (Uno, Nano, ou compatible)
- 3x LEDs (rouge, jaune, verte)
- 3x Résistances 220Ω
- Câbles de connexion
- (Optionnel) Breadboard

### Schéma de branchement

```
Arduino          Composants
────────         ──────────
Pin 10 ────[220Ω]────LED VERTE─────┐
Pin 9  ────[220Ω]────LED JAUNE─────┼──── GND
Pin 8  ────[220Ω]────LED ROUGE─────┘
```

### Téléverser le code Arduino

1. Ouvrir **Arduino IDE**
2. Ouvrir le fichier `arduino/led_controller.ino`
3. Sélectionner la carte : `Outils` → `Type de carte` → Votre Arduino
4. Sélectionner le port : `Outils` → `Port` → COM correspondant
5. Cliquer sur **Téléverser** (➡️)

### Comportement des LEDs

| Score de cohérence | LED allumée |
|--------------------|-------------|
| 70-100% | 🟢 Verte (cohérent) |
| 40-69% | 🟡 Jaune (moyen) |
| 0-39% | 🔴 Rouge (incohérent) |

---

## 🎯 Utilisation

1. **Ouvrir l'application** : http://localhost:3000
2. **Poser une question** :
   - Tapez dans la zone de texte, ou
   - Cliquez sur le micro pour enregistrer vocalement
3. **Recevoir la réponse** :
   - La réponse s'affiche avec un badge de cohérence
   - La réponse est lue à voix haute
   - (Si Arduino connecté) La LED correspondante s'allume

### Utilisation / configuration du port Arduino

Le backend tente d'auto-détecter un Arduino connecté. Si nécessaire, vous pouvez forcer le port COM en définissant la variable d'environnement `ARDUINO_PORT` (ex. `COM3` sous Windows).

1. Dans `server/.env` ajoutez par exemple :

```env
ARDUINO_PORT=COM3
```

2. Ou exportez la variable avant de lancer le serveur :

Windows PowerShell:
```powershell
$env:ARDUINO_PORT = 'COM3'
python app.py
```

Si aucun Arduino n'est trouvé l'application fonctionne en mode simulation et affiche l'état des LEDs dans la console.

---

## 🔧 Dépannage

### Erreur : "Module not found"

```bash
pip install -r server/requirements.txt
npm install
```

### Erreur : "HF_TOKEN not found"

Vérifiez que le fichier `server/.env` existe et contient votre token Hugging Face.

### Erreur : "Port 5000 already in use"

```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :5000
kill -9 <PID>
```

### Erreur : "CORS error"

Vérifiez que le serveur Flask est bien lancé sur le port 5000.

### L'audio ne fonctionne pas

- Vérifiez que votre navigateur a accès au microphone
- Vérifiez que la clé ElevenLabs est valide

### Les LEDs ne s'allument pas

1. Vérifiez que l'Arduino est bien connecté
2. Vérifiez le port COM dans le Gestionnaire de périphériques
3. Testez avec le Moniteur Série d'Arduino IDE (envoyez "75" pour tester)

---

## 📁 Structure du projet

```
assistant-vocal-ece-react/
├── arduino/
│   └── led_controller.ino    # Code Arduino pour les LEDs
├── public/
│   └── index.html
├── server/
│   ├── app.py                # Serveur Flask principal
│   ├── openai_api.py         # API Hugging Face (chat + cohérence)
│   ├── whisper_api.py        # API Hugging Face (transcription)
│   ├── elevenlabs_api.py     # API ElevenLabs (synthèse vocale)
│   ├── arduino_controller.py # Communication Arduino
│   ├── requirements.txt      # Dépendances Python
│   └── .env                  # Variables d'environnement (à créer)
├── src/
│   ├── App.jsx               # Composant principal React
│   ├── components/           # Composants UI
│   └── services/             # Services API frontend
├── package.json              # Dépendances Node.js
└── README.md                 # Ce fichier
```

---

## 📄 Licence

MIT License - Voir [LICENSE](LICENSE)

---

## 👥 Auteurs

Projet réalisé dans le cadre d'un projet scolaire à l'ECE Paris.
