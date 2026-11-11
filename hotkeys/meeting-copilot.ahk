; Meeting Copilot — Hotkeys globaux Windows (AutoHotkey v1)
; - Ctrl+Shift+Space => Proposer une réponse    (POST /api/ask)
; - Ctrl+Alt+L     => Toggle FR/EN              (POST /api/toggle-language)
; - Ctrl+Alt+S     => Start/Stop                (POST /api/toggle-running)
; L'état (langue, session en cours) est géré uniquement par le serveur :
; les raccourcis restent cohérents avec les actions faites depuis l'interface web.

#NoEnv
#SingleInstance Force
SendMode Input
SetWorkingDir %A_ScriptDir%

global BaseUrl := "http://127.0.0.1:8001"

^+Space::
  HttpPost(BaseUrl "/api/ask", "{}")
return

^!l::
  HttpPost(BaseUrl "/api/toggle-language", "{}")
return

^!s::
  HttpPost(BaseUrl "/api/toggle-running", "{}")
return

HttpPost(url, json) {
  try {
    whr := ComObjCreate("WinHttp.WinHttpRequest.5.1")
    whr.Open("POST", url, false)
    whr.SetRequestHeader("Content-Type", "application/json")
    whr.Send(json)
    if (whr.Status >= 400)
      TrayTip, Meeting Copilot, % "Erreur HTTP " whr.Status " sur " url, 3, 2
  }
  catch {
    TrayTip, Meeting Copilot, % "Serveur injoignable (" url ")", 3, 2
  }
}
