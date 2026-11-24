# Careter

**Makes you a faster typer.** Word suggestions, grey-text autocomplete and auto-correct in every Windows app, private on your PC, plus an AI assistant at your cursor.

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

- The top suggestion appears as **grey text right after the cursor**; **Tab** inserts it. Below the line, a vertical list shows up to **5 suggestions**: **↑ / ↓** move the highlight (the grey text follows), **Tab** inserts the highlighted one, **1–5** insert one directly. After you accept, it suggests the word after that.
- Fast: word-list suggestions show about 0.1 s after you pause; if the LLM finds better ones, the list updates in place. The grey text is skipped when there's already text after the cursor on that line, or when the app doesn't report where the cursor is.
- Keep typing or press **Esc** and the strip goes away. Number keys, arrows and Tab behave normally whenever the list isn't showing.
- **Fixes while you type:** if the word you're typing looks misspelled, the list also offers what you probably meant, marked ✎ ("craet" → *created*, "becuas" → *because*). Picking one replaces what you typed.
- **Auto-correct:** when you press space after a typo, it's fixed: famous typos ("teh" → "the", "adn" → "and"), one-letter mistakes ("recevie" → "receive"), and two-letter mistakes in words of 7+ letters ("definatly" → "definitely"), and up to three in words of 9+ letters ("aknoldgement" → "acknowledgement"). Fixes keep the first letter and aim only at common words. Press **Backspace** right after to put back what you typed; that word is then never corrected again (saved, encrypted, in `ngrams/user_words.dat`). It only touches lowercase words of 3+ letters that follow another word, so names, ACRONYMS, numbers, the first word in a box (usernames, passwords) and terminals are left alone. `CARET_AUTOCORRECT=0` turns it off.
- **Work words:** `vocab.py` lists about 380 everyday work and tech words (bug, portal, deploy, ticket, dashboard, invoice, standup, hotfix…). When you start typing one it ranks higher (at most 2 of the 5 slots), missing ones are added, and they're never auto-corrected. Add your own in **Settings → My words** (product names, tech terms, Manglish): always offered, never corrected.
- **Learns from you:** it counts the words you type and which word you type after which, and puts your own habits first (up to 3 of the 5 slots). A pair or a word completion counts after you've typed it twice. Words you use 3+ times are never auto-corrected. Stored only on your PC as single words and word pairs (never sentences) in `ngrams/user_model.dat`, saved every 30 s and **encrypted with your Windows login** (DPAPI): only your Windows account on this PC can read it, with no password to manage. `CARET_LEARN=0` turns learning off. Nothing is learned in terminals, from the first word in a box, or while prediction is off.
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

- Grey completion right after the caret (in the page's own font) plus a vertical list of 5 suggestions; **↑ / ↓** to choose, **Tab** inserts the highlighted one, **1–5** or a click picks directly.
- Auto-correct on space (same rules as the desktop: famous typos, 1–3 letter mistakes by word length). **Backspace** right after undoes it, and that word is never corrected again.
- Learns your words and word pairs. **Double-tap Ctrl**, or click the toolbar icon, turns it off and on.
- Ignores password, email, phone, number and login- or card-looking fields.
- **Works alongside the desktop app:** when Careter is running on the PC, this extension stays quiet and the desktop app (with the helper below) handles everything, so you never see two strips.

Install: `chrome://extensions` → **Developer mode** → **Load unpacked** → choose `careter\careter-chrome`. On first use it downloads the word lists once (about 10 MB, from norvig.com) and keeps a compact 1–2 MB version.

Privacy: everything runs in the browser. Your learned words stay in this browser profile (`chrome.storage.local`, not encrypted the way the desktop app's are). Nothing is sent anywhere, apart from the one-time word-list download and a check whether the desktop app is running on `127.0.0.1`.

For the Chrome Web Store you'd also add icons, put a prebuilt `model.json` in the folder (the extension uses it instead of downloading), and write a privacy policy.

## Privacy

Your data stays on your PC. Suggestions, auto-correct and everything Careter learns run locally and are never sent anywhere; what it learns is encrypted with your Windows login. The only thing that goes online is the **Ctrl+Alt+Space assistant**: when you use it, the text around your cursor is sent to the AI service you chose in Settings (OpenAI, Claude, or another), and nowhere else. On first start Careter downloads public word lists (and the small AI model, if enabled); nothing about you is uploaded.

## Settings

Press **Ctrl+Alt+S** (or open **Careter Settings** from the Start menu, or run `python caret_ai.py --settings`). The window (a light, modern theme) has five tabs:

- **Home**: status at a glance (suggestions, auto-correct, words learned, assistant model), quick links, and a short why-Careter.
- **Settings**: word suggestions, grey text, how many suggestions (3–5), auto-correct, suggestions in terminals, the local AI model, the **AI assistant** (model name, API key, and an optional API address for any OpenAI-compatible service such as OpenRouter, Groq or Ollama, with **Save & test**), learning (export / forget), and start-with-Windows. Saved in `settings.json` (AI assistant details in `.env`) and applied immediately; the local-AI switch after **Restart Careter**.
- **How to use**: a guide for new users: every key, auto-correct, turning it off, the assistant, learning.
- **Share**: ready-made posts for LinkedIn, X and Instagram with Copy and "Copy & open" (X opens with the post filled in).
- **Install & update**: install, update (stop → pull → restart), the browser extensions, handy commands, uninstall. Click a command to copy it.
- **Log**: what Careter is doing, live (never what you typed), with **Copy** to paste it when reporting a problem.

## Always running (Windows)

```powershell
.\.venv\Scripts\python caret_ai.py --autostart on       # start at sign-in + restart within 5 min if it stops
.\.venv\Scripts\python caret_ai.py --autostart status
.\.venv\Scripts\python caret_ai.py --autostart off      # back to manual
.\.venv\Scripts\python caret_ai.py --stop               # close the running copy (comes back if auto-start is on)
```

The installer turns auto-start on. It uses a Windows scheduled task named **Careter** (visible in Task Scheduler) running as you, with no admin rights, no time limit, and it keeps going on battery. While auto-start is on, running `python caret_ai.py` by hand just says "already running"; use `--stop` first if you want a console copy for debugging.

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
