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
if (-not ((Test-Path $EnvFile) -and (Select-String -Path $EnvFile -Pattern "OPENAI_API_KEY" -Quiet))) {
    Write-Host "Word suggestions, grey text and auto-correct work without any key."
    Write-Host "The Ctrl+Alt+Space AI assistant needs one (you can add it later: Ctrl+Alt+S > Settings > AI assistant)."
    $Key = Read-Host "Paste your OpenAI API key, or just press Enter to skip"
    if ($Key.Trim()) { Add-Content -Path $EnvFile -Value "OPENAI_API_KEY=$($Key.Trim())" }
    else {
        # remember the choice so re-running the installer doesn't ask again (Careter ignores # lines)
        Add-Content -Path $EnvFile -Value "# OPENAI_API_KEY skipped at install: add one in Settings > AI assistant"
        Write-Host "Skipped. Careter works without the assistant until you add a key."
    }
}

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
Write-Host "Careter installed and always running. Press Ctrl+Alt+Space anywhere."
