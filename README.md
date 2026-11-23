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
- **Auto-correct:** when you press space after a typo, it's fixed: famous typos ("teh" → "the", "adn" → "and"), one-letter mistakes ("recevie" → "receive"), and two-letter mistakes in words of 7+ letters ("definatly" → "definitely"), and up to three in words of 9+ letters ("aknoldgement" → "acknowledgement"). Fixes keep the first letter and aim only at common words. Press **Backspace** right after to put back what you typed; that word is then never corrected again (saved, encrypted, in `ngrams/user_words.dat`). It only touches lowercase words of 3+ letters that follow another word, so names, ACRONYMS, numbers, the first word in a box (usernames, passwords) and terminals are left alone. `CARET_AUTOCORRECT=0` turns it off.
- **Learns from you:** it counts the words you type and which word you type after which, and puts your own habits first (up to 3 of the 5 slots). A pair counts after you've typed it twice; a word completion after 3 uses. Words you use 3+ times are never auto-corrected. Stored only on your PC as single words and word pairs (never sentences) in `ngrams/user_model.dat`, saved every 30 s and **encrypted with your Windows login** (DPAPI): only your Windows account on this PC can read it, with no password to manage. `CARET_LEARN=0` turns learning off. Nothing is learned in terminals, from the first word in a box, or while prediction is off.
- **Double-tap Ctrl** (or Ctrl+Alt+P) turns prediction and auto-correct off and on, e.g. when you need to type numbers or Manglish. A label at the mouse shows which.

Two engines, both local and free (neither uses the OpenAI key). What you type stays in memory on your PC and is never logged. It never runs in password boxes, in terminals (`PREDICT_IN_TERMINALS`), or inside Careter's own bubble.

- **`ngram`: instant.** Counts of the most common English words and word pairs (Peter Norvig's lists from Google's Web 1T data, about 10 MB, downloaded once into `careter/ngrams/`). It completes the word you're typing and predicts the next word from the one before.
- **`llm` (default): smarter.** A small open-source model (`HuggingFaceTB/SmolLM2-135M`) in its own low-priority process (`pip install -r requirements-llm.txt`). The word lists load first and answer whenever the LLM is still loading, missing, or slower than `PREDICT_LLM_TIMEOUT` (0.6 s), so the strip always stays quick. Turning an LLM's next-*token* guesses into words is done in `predict.py`.

| Env var | Default | Purpose |
|---|---|---|
| `CARET_PREDICT` | `1` | `0` starts with prediction off |
| `CARET_PREDICT_ENGINE` | `llm` | `llm` (with word lists as backup) or `ngram` only |
| `CARET_PREDICT_MODEL` | `HuggingFaceTB/SmolLM2-135M` | For `llm`: any Hugging Face **base** (not chat) causal LM |

### Your learned data

```powershell
.\.venv\Scripts\python caret_ai.py --export-learned learned.json   # readable copy (delete it after)
.\.venv\Scripts\python caret_ai.py --forget-learned                 # erase everything it learned
```

### Browser helper for the desktop app (Chrome, Edge, Brave)

Browsers don't tell Windows where the caret is, so without help the strip appears at the mouse pointer. The small extension in `chrome-extension/` fixes that: it sends Careter the text just before the caret and the caret's position, so suggestions sit right under your text and use the real text.

Install once: open `chrome://extensions` (or `edge://extensions`), turn on **Developer mode**, click **Load unpacked**, and choose the `careter\chrome-extension` folder.

Privacy: it talks only to Careter on this PC (`127.0.0.1:47822`, which accepts only browser extensions), sends at most 300 characters before the caret and never anything after it, never page addresses, and never password boxes (it only says "password box", so Careter also skips auto-correct and learning there). It can't see inside Google Docs (drawn on a canvas); there the mouse fallback is used.

## Careter for Chrome (standalone extension)

`careter-chrome/` does word prediction inside the browser on its own, with no desktop app. It works on any computer with Chrome or Edge (Windows, Mac, Chromebook, Linux):

- 5 suggestions in a rounded strip under the caret on any website; **1–5** or **Tab** picks, or click.
- Auto-correct on space (same rules as the desktop: famous typos, 1–3 letter mistakes by word length). **Backspace** right after undoes it, and that word is never corrected again.
- Learns your words and word pairs. **Double-tap Ctrl**, or click the toolbar icon, turns it off and on.
- Ignores password, email, phone, number and login- or card-looking fields.
- **Works alongside the desktop app:** when Careter is running on the PC, this extension stays quiet and the desktop app (with the helper below) handles everything, so you never see two strips.

Install: `chrome://extensions` → **Developer mode** → **Load unpacked** → choose `careter\careter-chrome`. On first use it downloads the word lists once (about 10 MB, from norvig.com) and keeps a compact 1–2 MB version.

Privacy: everything runs in the browser. Your learned words stay in this browser profile (`chrome.storage.local`, not encrypted the way the desktop app's are). Nothing is sent anywhere, apart from the one-time word-list download and a check whether the desktop app is running on `127.0.0.1`.

For the Chrome Web Store you'd also add icons, put a prebuilt `model.json` in the folder (the extension uses it instead of downloading), and write a privacy policy.

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
