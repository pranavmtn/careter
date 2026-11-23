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

## What it reads

- **Text around the caret:** 10 characters before and after it (`CONTEXT_CHARS`). Read by moving the caret (Shift+←/→ ×10, copy, collapse back) with the clipboard restored; terminals use Windows UI Automation instead. Stays inside the current section (stops at a blank line).
- **Selection:** whatever you have selected.
- **No screenshots** for now (`USE_SCREENSHOTS = False`): the AI works only from the caret text or selection.

When the bubble opens it first says what it sees and offers 2–4 one-click next actions; you can also type or speak your own. Settings (`CONTEXT_CHARS`, `SECTION_BREAKS`, `AUTO_SUGGEST`) are at the top of `caret_ai.py`.

## Word prediction (Windows, local, no API key)

While you type anywhere, Careter suggests the next word, or the rest of the word you're typing, in a small strip at the caret:

- Up to **5 suggestions**, numbered. Press **1–5** to insert one (**Tab** also takes the first). After you accept, it suggests the word after that.
- Keep typing or press **Esc** and the strip goes away. Number keys and Tab behave normally whenever the strip isn't showing.
- **Double-tap Ctrl** (or Ctrl+Alt+P) turns prediction off and on, e.g. when you need to type numbers. A label at the mouse shows which.

Two engines, both local and free (neither uses the OpenAI key). What you type stays in memory on your PC and is never logged. It never runs in password boxes, in terminals (`PREDICT_IN_TERMINALS`), or inside Careter's own bubble.

- **`ngram`: instant.** Counts of the most common English words and word pairs (Peter Norvig's lists from Google's Web 1T data, about 10 MB, downloaded once into `careter/ngrams/`). It completes the word you're typing and predicts the next word from the one before.
- **`llm` (default): smarter.** A small open-source model (`HuggingFaceTB/SmolLM2-135M`) in its own low-priority process (`pip install -r requirements-llm.txt`). The word lists load first and answer whenever the LLM is still loading, missing, or slower than `PREDICT_LLM_TIMEOUT` (0.6 s), so the strip always stays quick. Turning an LLM's next-*token* guesses into words is done in `predict.py`.

| Env var | Default | Purpose |
|---|---|---|
| `CARET_PREDICT` | `1` | `0` starts with prediction off |
| `CARET_PREDICT_ENGINE` | `llm` | `llm` (with word lists as backup) or `ngram` only |
| `CARET_PREDICT_MODEL` | `HuggingFaceTB/SmolLM2-135M` | For `llm`: any Hugging Face **base** (not chat) causal LM |

In Chrome and other apps that don't report their caret position, the strip appears at the mouse pointer.

## Install (one command)

These clone Careter into your home folder, install dependencies, ask for your OpenAI key (saved to `careter/.env`), and set it to start at login.

**Windows** (PowerShell):
```powershell
irm https://raw.githubusercontent.com/pranavmtn/careter/main/install.ps1 | iex
```

**macOS / Linux:**
```bash
curl -fsSL https://raw.githubusercontent.com/pranavmtn/careter/main/install.sh | bash
```

## Manual setup

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
