# Meeting Copilot — Assistant d'entretien en temps réel

Outil local Windows pour t'aider pendant un entretien ou une discussion Teams :

- **Transcription temps réel** via Deepgram
  - **ME** : micro capturé par le navigateur
  - **OTHER** : audio système (Teams, Zoom, etc.) via WASAPI loopback
- **Gestion de meetings** : sauvegarde et chargement de l'historique des conversations
- **Profils personnels** : charge ton CV/infos pour des réponses contextualisées
- **Suggestions IA** via Google Gemini (`gemini-3.6-flash` recommandé) :
  - Meilleure réponse concise
  - Alternative plus détaillée
  - Question de clarification
  - Explication des termes techniques mentionnés
- **Mode speaker** : automatique (micro=ME, loopback=OTHER) ou manuel

## Prérequis

- Windows 10/11
- Python 3.11+ (requis par google-genai 3.x)
- Navigateur moderne (Chrome, Firefox, Edge)
- Clé API Deepgram (`DEEPGRAM_API_KEY`)
- Clé API Gemini (`GEMINI_API_KEY`), à créer gratuitement sur [Google AI Studio](https://aistudio.google.com/apikey)

## Installation

### 1. Créer l'environnement Python

```powershell
cd "chemin\vers\interview"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend\requirements.txt
```

### 2. Configurer les clés API

Copier `.env.example` vers `.env` et renseigner :

```env
DEEPGRAM_API_KEY=ta_clé_deepgram
GEMINI_API_KEY=ta_clé_gemini
GEMINI_MODEL=gemini-3.6-flash
```

> **Note** : `gemini-3.6-flash` (dernier Flash "Stable") est recommandé : rapide et fiable pour suivre les consignes. `gemini-3.8-flash` est plus récent mais pas encore "Stable" ; `gemini-3.1-pro-preview` (Pro, preview) est plus lent et plus cher. Si `GEMINI_MODEL` est vide, `gemini-3.6-flash` est utilisé.

> **Confidentialité** : avec l'offre gratuite de l'API Gemini, Google peut utiliser les prompts pour améliorer ses produits. Pour des entretiens réels, préfère une clé rattachée à un projet avec facturation activée.

### 3. Configurer le loopback audio

Rien à renseigner dans `.env` : le loopback (OTHER) est **détecté automatiquement** à partir du **périphérique de sortie par défaut de Windows**.

Avant de lancer une session, mets comme sortie par défaut (Paramètres Windows → Son → Sortie) le casque/haut-parleurs où tu entends Teams, et vérifie que Teams utilise bien ce même périphérique.

Pour diagnostiquer, tu peux lister les devices audio :

```powershell
python -m backend.audio_devices
```

> Cette commande sert uniquement au diagnostic, elle ne configure rien.

> Le micro n'a plus besoin d'être configuré ici — le navigateur te demandera quel micro utiliser.

## Lancer

Double-clique sur `start.bat` (vérifie `.venv`, `.env` et les dépendances, lance le serveur et ouvre le navigateur), ou en ligne de commande :

```powershell
.\.venv\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001
```

Puis ouvre dans ton navigateur :

```
http://127.0.0.1:8001/
```

## Utilisation

### Interface principale

| Élément | Description |
|---------|-------------|
| **Start/Stop** | Démarre ou arrête la transcription |
| **FR/EN** | Change la langue (Deepgram + LLM) |
| **Reset** | Efface la transcription du meeting actuel |
| **Meeting** | Sélectionne un meeting existant ou en crée un nouveau |
| **Profil** | Sélectionne ton profil (CV) pour contextualiser les réponses |
| **Mode Auto/Manuel** | Auto = mic→ME, loopback→OTHER. Manuel = tu choisis avec le bouton ME/OTHER |
| **Proposer une réponse** | Génère des suggestions basées sur la conversation |

### Profils

Crée des fichiers `.txt` dans le dossier `profiles/` avec tes informations (CV, compétences, projets). Le LLM utilisera ces infos pour personnaliser ses réponses **sans inventer de faux faits**.

Exemple : `profiles/benjamin_koffler.txt`

### Meetings

Les transcriptions sont sauvegardées automatiquement dans `meetings/` au format JSON. Tu peux :
- Charger un meeting précédent via le dropdown
- Créer un nouveau meeting
- Le meeting actuel est sauvegardé en temps réel

### Keywords (optionnel)

Pour améliorer la reconnaissance de mots spécifiques peu communs, ajoute-les dans `keywords.txt` (un par ligne). 

> **Attention** : Trop de keywords cause des faux positifs. Garde la liste très courte (5-10 mots max).

## Raccourcis globaux (AutoHotkey)

Installe [AutoHotkey](https://www.autohotkey.com/), puis lance :

```
hotkeys/meeting-copilot.ahk
```

| Raccourci | Action |
|-----------|--------|
| `Ctrl+Alt+Space` | Proposer une réponse |
| `Ctrl+Alt+L` | Toggle FR/EN |
| `Ctrl+Alt+S` | Start/Stop |

> Les raccourcis basculent en fonction de l'**état réel du serveur** (langue, session en cours) : ils restent cohérents avec les actions faites depuis l'interface web.

## Dépannage

### OTHER ne transcrit rien
- Vérifie que le périphérique de sortie **par défaut** de Windows est bien celui où tu entends Teams (le loopback est détecté à partir de celui-ci)
- Assure-toi que Teams utilise bien ce périphérique audio
- Lance `python -m backend.audio_devices` pour vérifier que le périphérique est visible

### Erreur WASAPI
- Utilise un device géré par WASAPI (Windows Audio Session API)
- Le script doit tourner sur Windows

### Le micro ne fonctionne pas
- Autorise l'accès au micro dans le navigateur
- Vérifie que le bon micro est sélectionné dans le popup du navigateur

### Les réponses LLM inventent des infos
- Utilise un modèle Gemini "Stable" récent (`gemini-3.6-flash`) ; pour plus de rigueur, essaie `gemini-3.1-pro-preview` (plus lent)
- Évite les anciens modèles `gemini-2.5-*` (dépréciés)
- Vérifie que ton profil contient les infos pertinentes

### Erreur `Missing env var: GEMINI_API_KEY`
- Renseigne `GEMINI_API_KEY` dans `.env` puis relance le serveur

## Structure du projet

```
interview/
├── backend/
│   ├── __init__.py         # Module Python
│   ├── app.py              # Serveur FastAPI principal
│   ├── deepgram_stream.py  # Client Deepgram WebSocket
│   ├── gemini_suggest.py   # Génération de suggestions (Gemini)
│   ├── audio_devices.py    # Liste les périphériques audio
│   └── requirements.txt
├── web/
│   └── index.html          # Interface utilisateur
├── profiles/               # Fichiers de profil (.txt)
├── meetings/               # Transcriptions sauvegardées (.json)
├── hotkeys/
│   └── meeting-copilot.ahk # Raccourcis AutoHotkey
├── keywords.txt            # Mots à booster (optionnel)
├── .env                    # Configuration (clés API, modèle, langue)
└── README.md
```

## Confidentialité

- Les transcriptions sont stockées localement dans `meetings/`
- L'audio n'est pas enregistré, seul le texte est sauvegardé
- Les données sont envoyées à Deepgram (transcription) et Google Gemini (suggestions)
- Vérifie le cadre légal/consentement selon ton contexte
