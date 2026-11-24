# Careter installer for Windows. In PowerShell:
#   irm https://raw.githubusercontent.com/pranavmtn/careter/main/install.ps1 | iex
$ErrorActionPreference = "Stop"

$Dir  = Join-Path $env:USERPROFILE "careter"
$Repo = "https://github.com/pranavmtn/careter"

if (-not (Get-Command git -ErrorAction SilentlyContinue))    { Write-Host "Please install Git first: winget install Git.Git"; return }
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { Write-Host "Please install Python 3.10+ first: winget install Python.Python.3.12"; return }

if (Test-Path (Join-Path $Dir ".git")) { git -C $Dir pull --ff-only }
else { git clone $Repo $Dir }

Set-Location $Dir
python -m venv .venv
& "$Dir\.venv\Scripts\python.exe" -m pip install --upgrade pip | Out-Null
& "$Dir\.venv\Scripts\python.exe" -m pip install -r requirements.txt

$EnvFile = Join-Path $Dir ".env"
if (-not (Test-Path $EnvFile)) { New-Item -ItemType File -Path $EnvFile | Out-Null }
# No API key needed to install: typing help works without one. The Ctrl+Alt+Space assistant's key
# is added later in Ctrl+Alt+S > Settings > AI assistant.

# Start at login, without a console window
$PyW = "$Dir\.venv\Scripts\pythonw.exe"
$Startup = [Environment]::GetFolderPath("Startup")
$Shell = New-Object -ComObject WScript.Shell
$Lnk = $Shell.CreateShortcut((Join-Path $Startup "Careter.lnk"))
$Lnk.TargetPath = $PyW
$Lnk.Arguments = "`"$Dir\caret_ai.py`""
$Lnk.WorkingDirectory = $Dir
$Lnk.Save()

# Start menu: "Careter Settings" opens the settings window of the running Careter
$Programs = [Environment]::GetFolderPath("Programs")
$Set = $Shell.CreateShortcut((Join-Path $Programs "Careter Settings.lnk"))
$Set.TargetPath = $PyW
$Set.Arguments = "`"$Dir\caret_ai.py`" --settings"
$Set.WorkingDirectory = $Dir
$Set.Save()

# Keep it running: a scheduled task starts it at login and brings it back within 5 minutes if it
# stops (replaces the shortcut above when Windows allows a login trigger). This also starts it now.
& "$Dir\.venv\Scripts\python.exe" "$Dir\caret_ai.py" --autostart on
if ($LASTEXITCODE -ne 0) { Start-Process $PyW -ArgumentList "`"$Dir\caret_ai.py`"" -WorkingDirectory $Dir }
Write-Host ""
Write-Host "Careter is installed and running. $([char]0x2713)" -ForegroundColor Green
Write-Host "Type anywhere and pause: suggestions appear under the line."
Write-Host ""
Write-Host "  Tab / Enter       insert the suggestion (the grey text)"
Write-Host "  Up / Down         choose another suggestion"
Write-Host "  1 - 5             pick a suggestion directly"
Write-Host "  Esc               hide the suggestions"
Write-Host "  Backspace         right after an auto-correct: undo it"
Write-Host "  Ctrl, Ctrl        double-tap: suggestions on / off"
Write-Host "  Ctrl+Alt+Space    AI assistant at your cursor (add your AI key in Settings first)"
Write-Host "  Ctrl+Alt+S        settings, help and your AI key"
