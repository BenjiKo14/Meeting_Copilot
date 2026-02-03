# Meeting Copilot — Assistant d'entretien en temps réel

Outil local Windows pour t'aider pendant un entretien ou une discussion Teams :

- **Transcription temps réel** via Deepgram
  - **ME** : micro capturé par le navigateur
  - **OTHER** : audio système (Teams, Zoom, etc.) via WASAPI loopback
- **Gestion de meetings** : sauvegarde et chargement de l'historique des conversations
- **Profils personnels** : charge ton CV/infos pour des réponses contextualisées
- **Suggestions IA** via OpenAI (gpt-4o recommandé) :
  - Meilleure réponse concise
  - Alternative plus détaillée
  - Question de clarification
  - Explication des termes techniques mentionnés
- **Mode speaker** : automatique (micro=ME, loopback=OTHER) ou manuel

## Prérequis

- Windows 10/11
- Python 3.10+
- Navigateur moderne (Chrome, Firefox, Edge)
- Clé API Deepgram (`DEEPGRAM_API_KEY`)
- Clé API OpenAI (`OPENAI_API_KEY`)

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
OPENAI_API_KEY=ta_clé_openai
OPENAI_MODEL=gpt-4o
```

> **Note** : `gpt-4o` est recommandé car il suit mieux les instructions et n'invente pas de faux faits. `gpt-4o-mini` est moins cher mais peut "halluciner".

### 3. Configurer le loopback audio

Lister les devices audio :

```powershell
python -m backend.audio_devices
```

Dans `.env`, renseigne `LOOPBACK_DEVICE` avec l'index ou le nom du périphérique de **sortie** audio (casque/haut-parleurs où tu entends Teams) :

```env
LOOPBACK_DEVICE=4
```

> Le micro n'a plus besoin d'être configuré ici — le navigateur te demandera quel micro utiliser.

## Lancer

```powershell
.\.venv\Scripts\uvicorn backend.app:app --host 127.0.0.1 --port 8001 --reload
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

## Dépannage

### OTHER ne transcrit rien
- Vérifie que `LOOPBACK_DEVICE` pointe vers le bon périphérique de sortie
- Assure-toi que Teams utilise bien ce périphérique audio

### Erreur WASAPI
- Utilise un device géré par WASAPI (Windows Audio Session API)
- Le script doit tourner sur Windows

### Le micro ne fonctionne pas
- Autorise l'accès au micro dans le navigateur
- Vérifie que le bon micro est sélectionné dans le popup du navigateur

### Les réponses LLM inventent des infos
- Utilise `gpt-4o` au lieu de `gpt-4o-mini`
- Vérifie que ton profil contient les infos pertinentes

## Structure du projet

```
interview/
├── backend/
│   ├── __init__.py         # Module Python
│   ├── app.py              # Serveur FastAPI principal
│   ├── deepgram_stream.py  # Client Deepgram WebSocket
│   ├── openai_suggest.py   # Génération de suggestions
│   ├── audio_devices.py    # Liste les périphériques audio
│   └── requirements.txt
├── web/
│   └── index.html          # Interface utilisateur
├── profiles/               # Fichiers de profil (.txt)
├── meetings/               # Transcriptions sauvegardées (.json)
├── hotkeys/
│   └── meeting-copilot.ahk # Raccourcis AutoHotkey
├── keywords.txt            # Mots à booster (optionnel)
├── .env                    # Configuration (clés API, devices)
└── README.md
```

## Confidentialité

- Les transcriptions sont stockées localement dans `meetings/`
- L'audio n'est pas enregistré, seul le texte est sauvegardé
- Les données sont envoyées à Deepgram (transcription) et OpenAI (suggestions)
- Vérifie le cadre légal/consentement selon ton contexte
