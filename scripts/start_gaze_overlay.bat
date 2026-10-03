@echo off
rem 視線オーバーレイを手動で起動する。引数はそのまま渡す(例: start_gaze_overlay.bat --debug-view)
cd /d "%~dp0.."
uv run gaze-overlay %*
rem エラーで終了したときはメッセージを読めるように止める
if errorlevel 1 pause
