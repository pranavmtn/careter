#!/usr/bin/env bash
# Careter installer for macOS and Linux.
#   curl -fsSL https://raw.githubusercontent.com/pranavmtn/careter/main/install.sh | bash
set -euo pipefail

DIR="$HOME/careter"
REPO="https://github.com/pranavmtn/careter"

command -v git >/dev/null || { echo "Please install git first."; exit 1; }
command -v python3 >/dev/null || { echo "Please install Python 3.10+ first."; exit 1; }

if [ -d "$DIR/.git" ]; then
  echo "Updating $DIR"; git -C "$DIR" pull --ff-only
else
  echo "Cloning into $DIR"; git clone "$REPO" "$DIR"
fi

cd "$DIR"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip >/dev/null
.venv/bin/pip install -r requirements.txt

if ! grep -q "OPENAI_API_KEY" .env 2>/dev/null; then
  echo "The AI assistant needs an API key (you can add it later in .env as OPENAI_API_KEY=...)."
  read -r -p "Paste your OpenAI API key, or press Enter to skip: " KEY </dev/tty
  if [ -n "$KEY" ]; then echo "OPENAI_API_KEY=$KEY" >> .env
  else echo "# OPENAI_API_KEY skipped at install: add OPENAI_API_KEY=... here later" >> .env; echo "Skipped."; fi
  chmod 600 .env
fi

PY="$DIR/.venv/bin/python"
case "$(uname -s)" in
  Darwin)
    PLIST="$HOME/Library/LaunchAgents/com.careter.app.plist"
    mkdir -p "$(dirname "$PLIST")"
    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.careter.app</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>$DIR/caret_ai.py</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><false/>
  <key>StandardErrorPath</key><string>$DIR/careter.log</string>
</dict></plist>
EOF
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo
    echo "macOS: allow Screen Recording, Accessibility and Microphone for Python in"
    echo "System Settings > Privacy & Security, then run: launchctl kickstart -k gui/\$(id -u)/com.careter.app"
    ;;
  Linux)
    mkdir -p "$HOME/.config/autostart"
    cat > "$HOME/.config/autostart/careter.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Careter
Exec=$PY $DIR/caret_ai.py
X-GNOME-Autostart-enabled=true
EOF
    [ "${XDG_SESSION_TYPE:-}" = "wayland" ] && echo "Warning: Wayland detected. Careter needs an X11 session."
    nohup "$PY" "$DIR/caret_ai.py" > "$DIR/careter.log" 2>&1 &
    ;;
esac

echo
echo "Careter installed and set to start at login. Press Ctrl+Alt+Space anywhere."
