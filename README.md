# Careter

An AI that lives at your cursor. Point at any spot on screen (a terminal, an ebook, an editor, a form), press a hotkey, and type or speak what you want. Careter screenshots what you see, reads the text around that spot, proposes actions, asks you to confirm, runs them, looks at the result and continues until the goal is done.

## How it works

1. Put your pointer somewhere and press **Ctrl+Alt+Space**.
2. A popup opens next to it. Type the goal or click 🎤 to say it.
3. The model (GPT-6 Luna by default) gets a screenshot with a red crosshair at your spot plus a zoomed crop.
4. It shows the planned actions (`type: git status`, `press: enter`) and waits for **Run** or **Stop**. Risky actions are flagged in red.
5. After running, it looks at the screen again (e.g. the terminal output) and proposes the next step, up to 12 steps.
6. For reading tasks ("explain this paragraph", "translate this"), it answers in the popup and types nothing.

**Esc** closes the popup. Moving the mouse into a screen corner aborts typing.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python caret_ai.py
```

| Env var | Default | Purpose |
|---|---|---|
| `CARET_AI_MODEL` | `gpt-6-luna` | Vision + reasoning model |
| `CARET_AI_STT` | `whisper-1` | Speech-to-text model for the mic |

### Platform notes
- **macOS:** grant your terminal Screen Recording, Accessibility and Microphone permission in System Settings.
- **Linux:** needs an X11 session (Wayland blocks global hotkeys and simulated input).
- **Windows:** works as-is.
