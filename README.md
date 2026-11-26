# Careter

**Makes you a faster typer.** Word suggestions, grey-text autocomplete and auto-correct in every Windows app, private on your PC, plus an AI assistant at your cursor.

An AI that lives at your cursor. Point at any spot on screen (a terminal, an ebook, an editor, a form), press a hotkey, and type or speak what you want. Careter screenshots what you see, reads the text around that spot, proposes actions, asks you to confirm, runs them, looks at the result and continues until the goal is done.

## How it works

The assistant reads about 100 words before and after the cursor (widened to whole sentences, stopping at a blank line once it has 40+ words) so it understands what you're writing.

1. Put your pointer somewhere and press **Ctrl+Alt+Space**.
2. A popup opens next to it. Type the goal or click 🎤 to say it.
3. The model (GPT-6 Luna by default) gets a screenshot with a red crosshair at your spot plus a zoomed crop.
4. It shows the planned actions (`type: git status`, `press: enter`) and waits for **Run** or **Stop**. Risky actions are flagged in red.
5. After running, it looks at the screen again (e.g. the terminal output) and proposes the next step, up to 12 steps.
6. For reading tasks ("explain this paragraph", "translate this"), it answers in the popup and types nothing.

**Esc** closes the popup. Moving the mouse into a screen corner aborts typing.

### Rewrite in your style: Ctrl+Alt+R

Select some text (or just put the cursor on a line) and press **Ctrl+Alt+R**. With no popup, it's replaced in place by a cleaner version that still sounds like you: same tone, same variety of English (Indian English stays Indian English), and Manglish or mixed languages stay mixed. **Ctrl+Z** brings your original back. Describe how you write in **Settings → Rewrite** (for example "Indian English, friendly, short sentences"). It also sends your 40 most used words so it picks the words you'd pick, and a switch there turns that off. It uses the same AI key as the assistant. If you switch windows while it's working, the rewrite goes to the clipboard instead of being pasted. If nothing happens (no AI key, nothing selected…), Settings → Log says why.

## What it reads

- **Text around the caret:** 10 characters before and after it (`CONTEXT_CHARS`). Read by moving the caret (Shift+←/→ ×10, copy, collapse back) with the clipboard restored; terminals use Windows UI Automation instead. Stays inside the current section (stops at a blank line).
- **Selection:** whatever you have selected.
- **No screenshots** for now (`USE_SCREENSHOTS = False`): the AI works only from the caret text or selection.

When the bubble opens it first says what it sees and offers 2–4 one-click next actions; you can also type or speak your own. Settings (`CONTEXT_CHARS`, `SECTION_BREAKS`, `AUTO_SUGGEST`) are at the top of `caret_ai.py`.

## Word prediction (Windows and Ubuntu/Linux, local, no API key)

While you type anywhere, Careter suggests the next word, or the rest of the word you're typing, in a small strip at the caret:

- The top suggestion appears as **grey text right after the cursor**; **Tab** or **Enter** inserts it. Below the line, a vertical list shows up to **5 suggestions**: **↑ / ↓** move the highlight (the grey text follows), **Tab** or **Enter** inserts the highlighted one, **1–5** insert one directly. After you accept, it suggests the word after that.
- Fast: word-list suggestions show about 0.1 s after you pause; if the LLM finds better ones, the list updates in place. The grey text is skipped when there's already text after the cursor on that line, or when the app doesn't report where the cursor is.
- **Mouse:** click a suggestion to insert it; press and drag the list (anywhere, or by its **• • •** grip) if it covers something; it stays there (relative to the cursor) in that window, and a double-click on the grip puts it back. **Enter** picking a suggestion can be switched off in Settings.
- Keep typing or press **Esc** and the strip goes away. Number keys, arrows, Tab and Enter behave normally whenever the list isn't showing (press Esc first for a plain new line).
- **Fixes while you type:** if the word you're typing looks misspelled, the list also offers what you probably meant, marked ✎ ("craet" → *created*, "becuas" → *because*). Picking one replaces what you typed.
- **Auto-correct:** when you press space after a typo, it's fixed: famous typos ("teh" → "the", "adn" → "and"), one-letter mistakes ("recevie" → "receive"), and two-letter mistakes in words of 7+ letters ("definatly" → "definitely"), and up to three in words of 9+ letters ("aknoldgement" → "acknowledgement"). Fixes keep the first letter and aim only at common words. Press **Backspace** right after to put back what you typed; that word is then never corrected again (saved, encrypted, in `ngrams/user_words.dat`). It only touches lowercase words of 3+ letters that follow another word, so names, ACRONYMS, numbers, the first word in a box (usernames, passwords) and terminals are left alone. `CARET_AUTOCORRECT=0` turns it off.
- **Work words:** `vocab.py` lists about 380 everyday work and tech words (bug, portal, deploy, ticket, dashboard, invoice, standup, hotfix…). When you start typing one it ranks higher (at most 2 of the 5 slots), missing ones are added, and they're never auto-corrected. Add your own in **Settings → My words** (product names, tech terms, Manglish): always offered, never corrected.
- **Manglish built in:** about 110 Manglish words and phrases (sheri, adipoli, machane, pinne kaanam, njan varaam, tension adikkanda…) are suggested as you type and never auto-corrected (so "illa" no longer becomes "ill"). After a Manglish word its phrase comes first ("pinne " → kaanam, paranja); otherwise one Manglish match takes the last slot so English typing isn't crowded out. Edit or extend the list in **Settings → Manglish** (comma-separated; **Reset to the starting list** brings the original back). The starting list lives in `vocab.py`; the Chrome extension has the same list built in. Needs learning on.
- **Your most used words first** (Settings → Learning, on by default): words and word pairs you've typed twice come first and may fill the list; Settings shows **Your most used words**.
- **Learns from you:** it counts the words you type and which word you type after which, and puts your own habits first (up to 3 of the 5 slots). A pair or a word completion counts after you've typed it twice. A word you type **instead of** the suggestions shown counts double, so it's suggested first the very next time (desktop app and Chrome extension). Words you use 3+ times are never auto-corrected. Stored only on your PC as single words and word pairs (never sentences) in `ngrams/user_model.dat`, saved every 30 s and **encrypted with your Windows login** (DPAPI): only your Windows account on this PC can read it, with no password to manage. `CARET_LEARN=0` turns learning off. Nothing is learned in terminals, from the first word in a box, or while prediction is off.
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

Your data stays on your PC. Suggestions, auto-correct and everything Careter learns run locally and are never sent anywhere; what it learns is encrypted with your Windows login. The only things that go online are the **Ctrl+Alt+Space assistant** and **Ctrl+Alt+R rewrite**: when you use them, the text around your cursor (for rewrite: the text you're rewriting, plus your 40 most used words unless you turn that off) is sent to the AI service you chose in Settings (OpenAI, Claude, or another), and nowhere else. On first start Careter downloads public word lists (and the small AI model, if enabled); nothing about you is uploaded.

## Settings

Press **Ctrl+Alt+S**, click the **⚙** on the suggestion list (handy if Ctrl+Alt+S is taken on your PC), or open **Careter Settings** from the Start menu ( or run `python caret_ai.py --settings`). The window (a light, modern theme) has five tabs:

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

No API key needed: the installer doesn't ask for one. Word suggestions, grey text, auto-correct and learning work without it; only the Ctrl+Alt+Space assistant and Ctrl+Alt+R rewrite need a key, which you add later in **Settings → AI assistant** (OpenAI, Claude or any OpenAI-compatible service). When it finishes, the installer lists the shortcuts, and Careter shows them once in a small card.

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
- **Linux:** needs an X11 session (Wayland blocks global hotkeys and simulated input). See *Ubuntu / Linux* below.
- **Windows:** works as-is.

## Ubuntu / Linux

Word suggestions, grey text, auto-correct, learning, Manglish, the assistant and rewrite all work on Ubuntu and other Linux desktops running X11.

1. **Use X11, not Wayland.** Log out, click the ⚙ at the bottom right of the login screen, choose **Ubuntu on Xorg**, log in. `echo $XDG_SESSION_TYPE` should print `x11`.
2. **Install:** `curl -fsSL https://raw.githubusercontent.com/pranavmtn/careter/main/install.sh | bash`. It installs the system packages it needs with apt (`python3-tk`, `python3-venv`, `xclip`, `libportaudio2`, `python3-gi`, `gir1.2-atspi-2.0`), so it may ask for your password. It also turns on GNOME's accessibility setting so apps report where the cursor is, sets Careter to start at login, and starts it.
3. **Update:** `~/careter/.venv/bin/python ~/careter/caret_ai.py --stop`, then `cd ~/careter && git pull`, then `nohup ~/careter/.venv/bin/python ~/careter/caret_ai.py >/dev/null 2>&1 &`.

How it differs from Windows:
- **Cursor position** comes from accessibility (AT-SPI). GNOME apps (Text Editor, gedit), LibreOffice, Firefox and most GTK/Qt apps report it, so the list sits under your text and the grey text shows. In Chrome, add the browser helper extension (`chrome-extension/`) for the same. Apps that report nothing get the list at the mouse, without grey text.
- **Tab, Enter, ↑ ↓ and 1–5** are taken by Careter only while the list is showing (an X11 key grab), exactly like on Windows.
- **Auto-correct** fixes the word just after the space reaches the app (X11 can't hold a single key back), so you may see the typo for a split second.
- **Learned words** are stored in a file only your user can read (not encrypted with a login key as on Windows).
- **Not on Linux:** the Windows-only scheduled task (`--autostart`); login start uses `~/.config/autostart/careter.desktop` instead. Terminals are skipped by default, as on Windows.
- If something's off, `~/careter/careter.log` (or Settings → Log) says why. It never contains what you typed.

