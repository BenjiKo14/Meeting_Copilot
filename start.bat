@echo off
chcp 65001 >nul
setlocal
REM Meeting Copilot - lance le serveur sur http://127.0.0.1:8001/
REM (port 8001 attendu par hotkeys\meeting-copilot.ahk)

cd /d "%~dp0"
set "PORT=8001"
set "URL=http://127.0.0.1:%PORT%/"
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [ERREUR] Environnement Python introuvable ^(.venv^).
  echo Cree-le avec :  python -m venv .venv  puis relance ce script.
  pause
  exit /b 1
)

if not exist ".env" (
  echo [ERREUR] Fichier .env introuvable. Copie .env.example vers .env et renseigne les cles.
  pause
  exit /b 1
)

findstr /r /c:"^GEMINI_API_KEY=..*" ".env" >nul || (
  echo [ATTENTION] GEMINI_API_KEY absent ou vide dans .env : les suggestions et le chat ne marcheront pas.
)
findstr /r /c:"^DEEPGRAM_API_KEY=..*" ".env" >nul || (
  echo [ATTENTION] DEEPGRAM_API_KEY absent ou vide dans .env : la transcription ne demarrera pas.
)

"%PY%" -c "import fastapi, uvicorn, websockets, pyaudiowpatch, numpy, scipy, dotenv, google.genai, pydantic" 2>nul || (
  echo Installation des dependances...
  "%PY%" -m pip install -r backend\requirements.txt || (
    echo [ERREUR] Echec de l'installation des dependances.
    pause
    exit /b 1
  )
)

echo.
echo Meeting Copilot demarre sur %URL%  ^(Ctrl+C pour arreter^)
echo.

REM Ouvre le navigateur une fois le serveur pret (petit delai)
start "" /b cmd /c "timeout /t 3 /nobreak >nul & start "" %URL%"

"%PY%" -m uvicorn backend.app:app --host 127.0.0.1 --port %PORT%

echo.
echo Serveur arrete.
pause
