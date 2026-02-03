; Meeting Copilot — Hotkeys globaux Windows (AutoHotkey v1)
; - Ctrl+Alt+Space => Proposer une réponse
; - Ctrl+Alt+L     => Toggle FR/EN
; - Ctrl+Alt+S     => Start/Stop

#NoEnv
#SingleInstance Force
SendMode Input
SetWorkingDir %A_ScriptDir%

global Lang := "fr"
global Running := 0

^!Space::
  HttpPost("http://127.0.0.1:8001/api/ask", "{}")
return

^!l::
  if (Lang = "en")
    Lang := "fr"
  else
    Lang := "en"
  HttpPost("http://127.0.0.1:8001/api/language", "{""language"":""" Lang """}")
return

^!s::
  if (Running = 1) {
    HttpPost("http://127.0.0.1:8001/api/stop", "{}")
    Running := 0
  } else {
    HttpPost("http://127.0.0.1:8001/api/start", "{}")
    Running := 1
  }
return

HttpPost(url, json) {
  whr := ComObjCreate("WinHttp.WinHttpRequest.5.1")
  whr.Open("POST", url, false)
  whr.SetRequestHeader("Content-Type", "application/json")
  whr.Send(json)
}
