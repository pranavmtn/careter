#!/usr/bin/env bash
# Careter installer for macOS and Linux.
#   curl -fsSL https://raw.githubusercontent.com/pranavmtn/careter/main/install.sh | bash
set -euo pipefail

DIR="$HOME/careter"
REPO="https://github.com/pranavmtn/careter"

command -v git >/dev/null || { echo "Please install git first."; exit 1; }
command -v python3 >/dev/null || { echo "Please install Python 3.10+ first."; exit 1; }

if [ "$(uname -s)" = "Linux" ]; then
  # Ubuntu / Debian: what Careter needs from the system (windows, clipboard, mic, accessibility)
  if command -v dpkg >/dev/null; then
    NEED=""
    for p in python3-venv python3-tk python3-dev xclip libportaudio2 python3-gi gir1.2-atspi-2.0; do
      dpkg -s "$p" >/dev/null 2>&1 || NEED="$NEED $p"
    done
    if [ -n "$NEED" ]; then
      echo "Installing system packages:$NEED (asks for your password)"
      sudo apt-get update -qq && sudo apt-get install -y $NEED
    fi
  fi
  if [ "${XDG_SESSION_TYPE:-}" = "wayland" ]; then
    echo
    echo "NOTE: you're in a Wayland session. Careter needs X11: log out, click the gear icon at the"
    echo "bottom right of the login screen, choose \"Ubuntu on Xorg\", and log in. Then it starts by itself."
    echo
  fi
  # lets apps tell Careter where the cursor is (for the list and the grey text)
  command -v gsettings >/dev/null && gsettings set org.gnome.desktop.interface toolkit-accessibility true 2>/dev/null || true
fi

if [ -d "$DIR/.git" ]; then
  echo "Updating $DIR"; git -C "$DIR" pull --ff-only
else
  echo "Cloning into $DIR"; git clone "$REPO" "$DIR"
fi

cd "$DIR"
if [ "$(uname -s)" = "Linux" ]; then
  python3 -m venv --system-site-packages .venv   # uses Ubuntu's accessibility module (python3-gi)
else
  python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip >/dev/null
.venv/bin/pip install -r requirements.txt

touch .env   # the AI assistant's key goes here later: OPENAI_API_KEY=...
chmod 600 .env

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
    "$PY" "$DIR/caret_ai.py" --stop >/dev/null 2>&1 || true
    nohup "$PY" "$DIR/caret_ai.py" > "$DIR/careter-start.log" 2>&1 &   # (careter.log is its own log)
    ;;
esac

echo
echo "Careter is installed and starts every time you log in. Just type anywhere."
echo
echo "Shortcuts:"
echo "  Tab or Enter      insert the suggestion (grey text)"
echo "  Up / Down         choose another suggestion      1-5  pick one directly"
echo "  Double-tap Ctrl   suggestions on/off (also Ctrl+Alt+P)"
echo "  Ctrl+Alt+Space    AI assistant at your cursor"
echo "  Ctrl+Alt+R        rewrite the selected text (or the line) in your style"
echo "  Ctrl+Alt+S        settings"
echo
echo "The AI assistant and rewrite need an API key: add it in Settings (Ctrl+Alt+S) -> AI assistant."
