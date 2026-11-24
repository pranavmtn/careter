"""
Caret AI - an assistant that lives at your pointer.

Put your mouse/caret anywhere (terminal, ebook, browser, editor), press the
hotkey, then type or speak what you want. It screenshots the screen with a
marker at that spot, reads the text around it, proposes actions, asks you to
confirm, runs them, looks at the result, and keeps going until done.

Hotkey:  Ctrl+Alt+Space   (change HOTKEY below). Double-tap Ctrl: word prediction on/off.
Abort:   Esc in the popup, or slam the mouse into a screen corner while it types.
"""
import base64
import io
import json
import os
import platform
import queue
import re
import traceback
import socket
import sys
import tempfile
import threading
import time
import wave
import tkinter as tk

import mss
import numpy as np
import pyautogui
import pyperclip
import sounddevice as sd
from openai import OpenAI
from PIL import Image, ImageDraw
from pynput import keyboard, mouse

# ---------------------------------------------------------------- settings
def load_env_file():
    """Read KEY=VALUE lines from a .env next to this script (written by the installers)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env_file()
MODEL = os.getenv("CARET_AI_MODEL", "gpt-6-luna")
TRANSCRIBE_MODEL = os.getenv("CARET_AI_STT", "whisper-1")
HOTKEY = "<ctrl>+<alt>+<space>"
MAX_STEPS = 12          # max look -> act cycles per request
FOCUS_CLICK = True      # click the chosen spot before typing, to give it focus back
SAMPLE_RATE = 16000
CONTEXT_WORDS = 100     # words the assistant reads before and after the caret (widened to whole sentences)
CONTEXT_CHARS = 1200    # how far to look for them
USE_SCREENSHOTS = False # text-only for now: the AI reads only the text at the caret
DEBUG = True            # show the live log inside the bubble (toggle with 🐞)
SECTION_BREAKS = True   # stop at a blank line (end of a section), once at least 40 words are read
AUTO_SUGGEST = True     # when the bubble opens, analyse the spot and offer next actions

# Word prediction while you type (Windows). Runs a small local model: no API key, nothing leaves the PC.
PREDICT = os.getenv("CARET_PREDICT", "1") != "0"
PREDICT_ENGINE = os.getenv("CARET_PREDICT_ENGINE", "llm")    # "llm" (smarter) or "ngram" (word counts only)
PREDICT_LLM_TIMEOUT = 0.6     # if the LLM takes longer, show the word-count suggestions instead
PREDICT_MODEL = os.getenv("CARET_PREDICT_MODEL", "HuggingFaceTB/SmolLM2-135M")   # llm engine: a *base* model
PREDICT_DELAY = 0.06          # seconds of no typing before suggestions appear
PREDICT_COUNT = 5             # suggestions shown, picked with keys 1-5
PREDICT_IN_TERMINALS = False  # shell commands aren't prose; off by default
PREDICT_HOTKEY = "double-tap Ctrl (or Ctrl+Alt+P)"  # turns predictions (and auto-correct) on/off
AUTOCORRECT = os.getenv("CARET_AUTOCORRECT", "1") != "0"   # fix clear typos when you press space
BROWSERS = {"chrome.exe", "msedge.exe", "brave.exe", "vivaldi.exe", "opera.exe", "chromium.exe"}
WEB_PORT = 47822              # the Careter browser extension reports the caret here (127.0.0.1 only)
LEARN = os.getenv("CARET_LEARN", "1") != "0"   # learn your words & word pairs (ngrams/user_model.dat, encrypted)
GHOST = True                  # grey completion right after the cursor
ENTER_PICKS = True            # Enter inserts the highlighted suggestion (like Tab) while the list shows
USER_NAME = ""                # your name (Settings): the greeting, and the assistant signs drafts with it
WELCOMED = False              # the shortcuts card was shown once (first start)

# Settings window (Ctrl+Alt+S) choices are saved here and win over the defaults above.
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
SETTINGS = {"predict": "PREDICT", "ghost": "GHOST", "count": "PREDICT_COUNT", "autocorrect": "AUTOCORRECT",
            "learn": "LEARN", "engine": "PREDICT_ENGINE", "terminals": "PREDICT_IN_TERMINALS",
            "name": "USER_NAME", "welcomed": "WELCOMED", "enter": "ENTER_PICKS"}


def load_settings():
    import json
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return
    for key, name in SETTINGS.items():
        if key in data and type(data[key]) is type(globals()[name]):
            globals()[name] = data[key]


def save_settings():
    import json
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump({key: globals()[name] for key, name in SETTINGS.items()}, f, indent=1)


load_settings()

IS_MAC = platform.system() == "Darwin"
IS_WIN = platform.system() == "Windows"
# Never send Ctrl+C to these (it would interrupt the running command)
TERMINALS = {"windowsterminal.exe", "wt.exe", "powershell.exe", "pwsh.exe", "cmd.exe",
             "conhost.exe", "mintty.exe", "openconsole.exe", "alacritty.exe", "wezterm-gui.exe"}
pyautogui.FAILSAFE = True   # move mouse to a screen corner to abort typing
def has_ai_key():
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


# The AI assistant (Ctrl+Alt+Space) needs a key; everything for typing works without one.
client = OpenAI() if has_ai_key() else None

SUGGEST_SYSTEM = """You help a user who just called an assistant at their text caret.
You get the text around the caret ("▮" marks the caret, [[ ]] marks a selection; this is only the
section the caret is in). You do not see the screen, only this text.
1. In one short line, say what this is (e.g. "Bullet list of essay topics", "PowerShell after a failed pip install").
2. Offer 2-4 next actions the user most likely wants right here, as button labels of at most 5 words
   (e.g. "Fix grammar", "Continue writing", "Explain this", "Fix the error").
Reply ONLY with JSON: {"summary": "...", "suggestions": ["...", "..."]}"""

SYSTEM = f"""You are Caret AI, an assistant that lives at the user's pointer on their {platform.system()} desktop.
Each turn you receive the user's goal, the app they are in, and the TEXT AROUND THE CARET
(a few characters before and after it) or the text they selected. You do NOT see the screen;
work only from that text and the goal. If the text is too little to act on, ask a short question in "say".

Reply with ONLY a JSON object, no markdown:
{{
  "say": "short message to the user: the answer, an explanation, or what you are about to do",
  "actions": [],
  "risky": false,
  "done": false
}}

Action types:
  {{"type": "type", "text": "..."}}            types text at the spot (no Enter unless you add a key action)
  {{"type": "key", "keys": ["enter"]}}         presses a key or combo, e.g. ["ctrl","c"], ["tab"], ["up"]
  {{"type": "wait", "seconds": 2}}             waits (e.g. for a command to finish) before the next look

Rules:
- Terminal: type the command, press enter, then wait. You may not see the output
  afterwards, so prefer one complete command; set done true unless you are sure what comes next.
- If the user only wants information (explain, translate, summarize, define what is on screen), put the
  answer in "say", leave "actions" empty and set "done": true.
- Set "risky": true for anything destructive or hard to undo (deleting files, sudo, force push, killing
  processes, installing/uninstalling, sending messages, payments).
- Set "done": true when the goal is achieved or you cannot proceed, and say why.
- TEXT AROUND CARET: "▮" marks the caret and [[ ]] marks the selection.
- To insert text at the caret, use a "type" action with just the new text.
- If SELECTED TEXT is given, that is what the user means by "this", "the text", "selected". To change it,
  return ONE "type" action containing the full replacement text: typing replaces the selection. No enter.
- Never pretend to know text you were not given. Keep each turn's action list short.
"""


# ---------------------------------------------------------------- screen
def capture(anchor):
    """Screenshot the monitor under the pointer, mark the pointer, return (full, crop) as JPEG data URLs."""
    x, y = anchor
    with getattr(mss, "MSS", mss.mss)() as sct:
        mon = next(
            (m for m in sct.monitors[1:]
             if m["left"] <= x < m["left"] + m["width"] and m["top"] <= y < m["top"] + m["height"]),
            sct.monitors[1],
        )
        raw = sct.grab(mon)
    img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

    scale = img.width / mon["width"]          # handles Retina / HiDPI
    px, py = (x - mon["left"]) * scale, (y - mon["top"]) * scale

    draw = ImageDraw.Draw(img)
    r, w = 18 * scale, max(2, int(3 * scale))
    draw.ellipse([px - r, py - r, px + r, py + r], outline=(255, 0, 0), width=w)
    draw.line([px - 2 * r, py, px + 2 * r, py], fill=(255, 0, 0), width=w)
    draw.line([px, py - 2 * r, px, py + 2 * r], fill=(255, 0, 0), width=w)

    cw, ch = 900 * scale, 450 * scale
    left = max(0, min(px - cw / 2, img.width - cw))
    top = max(0, min(py - ch / 2, img.height - ch))
    crop = img.crop((int(left), int(top), int(left + cw), int(top + ch)))

    full = img.copy()
    full.thumbnail((1600, 1600))
    return to_data_url(full), to_data_url(crop)


def to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- actions
KEY_ALIASES = {"return": "enter", "esc": "escape", "cmd": "command" if IS_MAC else "win",
               "control": "ctrl", "option": "alt", "del": "delete"}


# Windows: send keys ourselves. pyautogui sends arrows/Home/End as *numpad* keys, and with Num Lock on
# Windows cancels Shift for those, so Shift+Left moves the caret without selecting anything.
VK = {"shift": 0x10, "ctrl": 0x11, "alt": 0x12, "win": 0x5B, "command": 0x5B,
      "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28, "home": 0x24, "end": 0x23,
      "pageup": 0x21, "pagedown": 0x22, "pgup": 0x21, "pgdn": 0x22, "insert": 0x2D, "delete": 0x2E,
      "enter": 0x0D, "tab": 0x09, "escape": 0x1B, "backspace": 0x08, "space": 0x20}
EXTENDED_VK = {0x25, 0x26, 0x27, 0x28, 0x24, 0x23, 0x21, 0x22, 0x2D, 0x2E, 0x5B}


def _vk(key):
    key = key.lower()
    if key in VK:
        return VK[key]
    if re.fullmatch(r"f([1-9]|1[0-2])", key):
        return 0x6F + int(key[1:])
    if len(key) == 1:
        import ctypes
        code = ctypes.windll.user32.VkKeyScanW(ctypes.c_wchar(key))
        return code & 0xFF if code != -1 else None
    return None


def _win_key(vk, up):
    import ctypes
    user32 = ctypes.windll.user32
    flags = (0x0001 if vk in EXTENDED_VK else 0) | (0x0002 if up else 0)   # EXTENDEDKEY | KEYUP
    user32.keybd_event(vk, user32.MapVirtualKeyW(vk, 0), flags, 0)


def send_hotkey(*keys):
    """Press keys together (e.g. send_hotkey('shift', 'left')), real arrow keys on Windows."""
    keys = [KEY_ALIASES.get(k.lower(), k.lower()) for k in keys]
    vks = [_vk(k) for k in keys] if IS_WIN else []
    if IS_WIN and all(vks):
        for v in vks:
            _win_key(v, False)
        for v in reversed(vks):
            _win_key(v, True)
        time.sleep(0.005)
    else:
        pyautogui.hotkey(*keys)


def paste(text):
    old = pyperclip.paste()
    pyperclip.copy(text)
    send_hotkey("command" if IS_MAC else "ctrl", "v")
    time.sleep(0.2)
    pyperclip.copy(old)


def wait_for_hotkey_release(timeout=1.5):
    """Don't send keys while the user still holds the hotkey keys."""
    if IS_WIN:
        import ctypes
        user32 = ctypes.windll.user32
        end = time.time() + timeout
        while time.time() < end and any(user32.GetAsyncKeyState(vk) & 0x8000 for vk in (0x11, 0x12, 0x20)):
            time.sleep(0.02)
        time.sleep(0.05)
    else:
        time.sleep(0.4)


def foreground_window():
    """(window handle, process name) of the app the user is in. Windows only."""
    if not IS_WIN:
        return None, ""
    import ctypes
    from ctypes import wintypes
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    hwnd = user32.GetForegroundWindow()
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    name = ""
    handle = kernel32.OpenProcess(0x1000, False, pid.value)
    if handle:
        buf, size = ctypes.create_unicode_buffer(260), wintypes.DWORD(260)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            name = os.path.basename(buf.value).lower()
        kernel32.CloseHandle(handle)
    return hwnd, name


LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "careter.log")


LOG_LINES = []          # in-memory log, shown in the bubble's debug panel
LOG_LISTENERS = []      # callbacks (bubbles) that want new lines


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    LOG_LINES.append(line)
    del LOG_LINES[:-300]
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    for fn in list(LOG_LISTENERS):
        try:
            fn(line)
        except Exception:
            pass


def run_with_timeout(fn, seconds, default=None):
    """UI Automation can hang on a frozen app; never let it block the hotkey."""
    box = [default]

    def target():
        try:
            box[0] = fn()
        except Exception:
            pass
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    return box[0]


def read_text_context(chars=CONTEXT_CHARS):
    """(before, selected, after) around the caret, read from the focused control with Windows
    UI Automation. Only the focused element (the box/div you are typing in) is read, so other
    blocks on the page are ignored. If the caret is on an empty new line/block, nothing above it
    is read. Returns None when the app doesn't expose its text (then we rely on the screenshot)."""
    if not IS_WIN:
        return None
    import uiautomation as auto
    with auto.UIAutomationInitializerInThread():
        ctrl, tp = auto.GetFocusedControl(), None
        for _ in range(6):                    # the focused node may be a child of the text box
            if not ctrl:
                break
            tp = ctrl.GetPattern(auto.PatternId.TextPattern)
            if tp:
                break
            ctrl = ctrl.GetParentControl()
        sel = tp.GetSelection() if tp else None
        if not sel:
            return None
        caret = sel[0]
        START, END = auto.TextPatternRangeEndpoint.Start, auto.TextPatternRangeEndpoint.End
        selected = caret.GetText(4000)

        b = caret.Clone()
        b.MoveEndpointByRange(END, caret, START)              # collapse to the caret start
        b.MoveEndpointByUnit(START, auto.TextUnit.Character, -chars)
        before = b.GetText(chars * 4)
        a = caret.Clone()
        a.MoveEndpointByRange(START, caret, END)              # collapse to the caret end
        a.MoveEndpointByUnit(END, auto.TextUnit.Character, chars)
        after = a.GetText(chars * 4)

    return trim_context(before, selected, after, chars)


def read_before_caret(chars=300):
    """For word prediction: (text before the caret, caret screen point (x, top, bottom) or None,
    is_password). Read-only (never moves the caret). None if the app doesn't expose its text."""
    if not IS_WIN:
        return None
    import uiautomation as auto
    with auto.UIAutomationInitializerInThread():
        return _read_before(auto, chars, {})


def _read_before(auto, chars, cache):
    """read_before_caret's work, on a thread that already set up UI Automation. `cache` remembers
    which element held the text for the focused control, so the next read skips the search."""
    ctrl, tp = auto.GetFocusedControl(), None
    if ctrl is None:
        return None
    try:
        if ctrl.IsPassword:
            return "", None, True
    except Exception:
        pass
    try:
        key = tuple(ctrl.GetRuntimeId() or ())
    except Exception:
        key = None
    if key and cache.get("key") == key:
        tp = cache.get("tp")
    else:
        for _ in range(6):
            if not ctrl:
                break
            tp = ctrl.GetPattern(auto.PatternId.TextPattern)
            if tp:
                break
            ctrl = ctrl.GetParentControl()
        cache.clear()
        if key and tp:
            cache.update(key=key, tp=tp)
    sel = tp.GetSelection() if tp else None
    if not sel:
        return None
    caret = sel[0]
    START, END = auto.TextPatternRangeEndpoint.Start, auto.TextPatternRangeEndpoint.End
    b = caret.Clone()
    b.MoveEndpointByRange(END, caret, START)
    b.MoveEndpointByUnit(START, auto.TextUnit.Character, -chars)
    text = b.GetText(chars * 4).replace("\r\n", "\n").replace("\r", "\n")
    point = None
    try:
        rects = b.GetBoundingRectangles()
        if rects:
            r = rects[-1]                          # the line the caret is on
            point = (r.right, r.top, r.bottom)
    except Exception:
        pass
    return text, point, False


class UiaReader:
    """One long-lived UI Automation thread for word prediction. Setting UIA up costs time, so it's
    done once instead of on every pause; only the newest request is worked on, and a frozen app can
    only make a read time out (then the typed keys are used)."""

    def __init__(self):
        self.jobs = queue.Queue()
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        import uiautomation as auto
        cache = {}
        with auto.UIAutomationInitializerInThread():
            while True:
                job = self.jobs.get()
                while not self.jobs.empty():           # skip requests you've typed past
                    job[2].set()
                    job = self.jobs.get()
                chars, box, done = job
                try:
                    box[0] = _read_before(auto, chars, cache)
                except Exception:
                    cache.clear()
                    box[0] = None
                done.set()

    def read(self, chars=300, timeout=0.25):
        box, done = [None], threading.Event()
        self.jobs.put((chars, box, done))
        done.wait(timeout)
        return box[0]


def around_words(text, words, side):
    """About `words` words of `text` next to the caret, widened to whole sentences (not strict)."""
    spans = list(re.finditer(r"\S+", text))
    if len(spans) <= words:
        return text
    if side == "before":
        cut = spans[-words].start()
        start = max((m.end() for m in re.finditer(r"[.!?\n]", text[max(0, cut - 300):cut])), default=None)
        return text[max(0, cut - 300) + start:] if start is not None else text[cut:]
    cut = spans[words - 1].end()
    end = re.search(r"[.!?\n]", text[cut:cut + 300])
    return text[:cut + end.end()] if end else text[:cut]


def trim_context(before, selected, after, chars=CONTEXT_CHARS):
    """About CONTEXT_WORDS words each side of the caret, in whole sentences. A blank line (a new
    section) stops the reading only once at least 40 words are in hand."""
    before, after = before.replace("\r\n", "\n"), after.replace("\r\n", "\n")
    if SECTION_BREAKS:
        parts, kept = re.split(r"(\n[ \t]*\n)", before), ""
        while parts and len(kept.split()) < 40:
            kept = parts.pop() + kept
        before = kept
        parts, kept = re.split(r"(\n[ \t]*\n)", after), ""
        while parts and len(kept.split()) < 40:
            kept = kept + parts.pop(0)
        after = kept
    before, after = around_words(before[-chars:], CONTEXT_WORDS, "before"), around_words(after[:chars], CONTEXT_WORDS, "after")
    if not (before.strip() or selected.strip() or after.strip()):
        return None
    return before, selected, after


def read_context_by_keys(proc, chars=CONTEXT_CHARS):
    """Fallback for apps that don't share their text (browser editors, Electron apps):
    Shift+Left xN, copy, Right (caret back); Shift+Right xN, copy, Left (caret back).
    Clipboard is restored. Never used in terminals, or when text is already selected."""
    if (IS_WIN and proc in TERMINALS) or not (IS_WIN or IS_MAC):
        return None
    mod = "command" if IS_MAC else "ctrl"
    try:
        old = pyperclip.paste()
    except Exception:
        old = ""

    def copy_side(extend_key, collapse_key):
        # one keypress selects to the start / end (Ctrl+Shift+Home / End); a plain arrow then puts
        # the caret back where it was
        sentinel = f"__careter_{time.time()}__"
        pyperclip.copy(sentinel)
        send_hotkey(mod, "shift", extend_key)
        time.sleep(0.05)
        send_hotkey(mod, "c")
        time.sleep(0.15)
        text = pyperclip.paste()
        log(f"  {mod}+shift+{extend_key} copied: {'nothing' if text == sentinel else f'{len(text)} chars'}")
        if text != sentinel and text:
            send_hotkey(collapse_key)        # collapse the selection back onto the caret
            time.sleep(0.03)
            return text
        return ""                             # at the start/end: nothing selected, caret didn't move

    try:
        before = copy_side("up" if IS_MAC else "home", "right")      # Mac: Cmd+Shift+Up = to the start
        after = copy_side("down" if IS_MAC else "end", "left")
    finally:
        pyperclip.copy(old)
    return trim_context(before, "", after)


def read_caret_context(proc):
    """Text around the caret. Normal apps: move the caret (Shift+arrows, copy, collapse back).
    Terminals (where those keys are unsafe) or if that fails: ask the app via UI Automation."""
    ctx = None
    if not (IS_WIN and proc in TERMINALS):
        try:
            ctx = read_context_by_keys(proc)
            log(f"keys -> {format_context(ctx) if ctx else 'nothing'}")
        except Exception:
            log("keys failed:\n" + traceback.format_exc())
    if not ctx and IS_WIN:
        ctx = run_with_timeout(read_text_context, 1.5)
        log(f"uia -> {format_context(ctx) if ctx else 'nothing'}")
    return ctx


def format_context(ctx):
    before, selected, after = ctx
    middle = f"[[{selected}]]" if selected else "▮"
    return f"{before}{middle}{after}"


def caret_rect():
    """Screen rect (left, top, bottom) of the text caret in the active app, or None. Windows only;
    many apps (browsers, Electron) don't expose a caret, so callers fall back to the mouse."""
    if not IS_WIN:
        return None
    import ctypes
    from ctypes import wintypes

    class GUITHREADINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD),
                    ("hwndActive", wintypes.HWND), ("hwndFocus", wintypes.HWND),
                    ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND),
                    ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND),
                    ("rcCaret", wintypes.RECT)]
    user32 = ctypes.windll.user32
    info = GUITHREADINFO(cbSize=ctypes.sizeof(GUITHREADINFO))
    tid = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)
    if not user32.GetGUIThreadInfo(tid, ctypes.byref(info)) or not info.hwndCaret:
        return None
    r = info.rcCaret
    if r.bottom - r.top <= 0:
        return None
    top_left = wintypes.POINT(r.left, r.top)
    bottom = wintypes.POINT(r.left, r.bottom)
    user32.ClientToScreen(info.hwndCaret, ctypes.byref(top_left))
    user32.ClientToScreen(info.hwndCaret, ctypes.byref(bottom))
    return top_left.x, top_left.y, bottom.y


def window_title(hwnd):
    if not (IS_WIN and hwnd):
        return ""
    import ctypes
    user32 = ctypes.windll.user32
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def monitor_bounds(x, y):
    with getattr(mss, "MSS", mss.mss)() as sct:
        m = next((m for m in sct.monitors[1:]
                  if m["left"] <= x < m["left"] + m["width"] and m["top"] <= y < m["top"] + m["height"]),
                 sct.monitors[1])
    return m["left"], m["top"], m["left"] + m["width"], m["top"] + m["height"]


def focus_window(hwnd):
    import ctypes
    user32 = ctypes.windll.user32
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)
    user32.SetForegroundWindow(hwnd)


def grab_selection(proc):
    """Copy the current selection via the clipboard, then restore the clipboard. '' if nothing selected."""
    if IS_WIN and proc in TERMINALS:
        return ""
    if not (IS_WIN or IS_MAC):      # on Linux Ctrl+C in a terminal would kill the command
        return ""
    try:
        old = pyperclip.paste()
    except Exception:
        old = ""
    sentinel = f"__careter_{time.time()}__"
    pyperclip.copy(sentinel)
    send_hotkey("command" if IS_MAC else "ctrl", "c")
    time.sleep(0.3)
    selected = pyperclip.paste()
    pyperclip.copy(old)
    return "" if selected == sentinel else selected


def run_actions(actions, anchor, click_to_focus=FOCUS_CLICK):
    if click_to_focus:
        pyautogui.click(*anchor)
        time.sleep(0.25)
    for a in actions:
        kind = a.get("type")
        if kind == "type":
            text = a.get("text", "")
            if text.isascii():
                pyautogui.write(text, interval=0.008)
            else:
                paste(text)
        elif kind == "key":
            keys = [KEY_ALIASES.get(k.lower(), k.lower()) for k in a.get("keys", [])]
            if keys:
                send_hotkey(*keys)
        elif kind == "wait":
            time.sleep(min(float(a.get("seconds", 1)), 60))
        time.sleep(0.1)
    time.sleep(0.8)   # let the screen settle before the next look


def describe(a):
    kind = a.get("type")
    if kind == "type":
        return f"type: {a.get('text', '')}"
    if kind == "key":
        return "press: " + "+".join(a.get("keys", []))
    if kind == "wait":
        return f"wait {a.get('seconds', 1)}s"
    return json.dumps(a)


# ---------------------------------------------------------------- model
def parse_reply(raw):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            data.setdefault("actions", [])
            return data
        except json.JSONDecodeError:
            pass
    return {"say": raw or "(empty reply)", "actions": [], "done": True}


def strip_old_images(msgs):
    """Keep only the latest screenshots in history to save tokens."""
    for m in msgs:
        if m["role"] == "user" and isinstance(m["content"], list):
            m["content"] = [c for c in m["content"] if c["type"] == "text"] + \
                           [{"type": "text", "text": "[older screenshot removed]"}]


# ---------------------------------------------------------------- audio
class Recorder:
    def __init__(self):
        self.frames, self.stream = [], None

    def start(self):
        self.frames = []
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                     callback=lambda data, *_: self.frames.append(data.copy()))
        self.stream.start()

    def stop_and_transcribe(self):
        self.stream.stop()
        self.stream.close()
        self.stream = None
        if not self.frames:
            return ""
        audio = np.concatenate(self.frames)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(audio.tobytes())
            with open(path, "rb") as f:
                return client.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=f).text
        finally:
            os.remove(path)


# ---------------------------------------------------------------- UI (tooltip bubble)
BG, FIELD, FG, MUTED, ACCENT, RED = "#1e1b2e", "#2a2640", "#f4f4f5", "#a1a1aa", "#8b5cf6", "#f87171"
FONT = ("Segoe UI", 10) if IS_WIN else ("Helvetica", 12)
FONT_B = (FONT[0], FONT[1], "bold")
WRAP = 380
# word-prediction strip: a rounded slate-grey pill with white words
STRIP_BG, STRIP_BORDER, STRIP_FG, STRIP_NUM = "#2b2e35", "#4b505a", "#ffffff", "#8f96a3"
STRIP_SEL = "#41464f"         # the highlighted row (Tab / Enter-free pick, moved with up/down)
GHOST_FG = "#9aa0a6"          # grey completion shown right after the cursor
# settings window: a light, modern theme (white, near-black text, grey details)
S_BG, S_FG, S_MUTED, S_LINE = "#ffffff", "#111827", "#6b7280", "#e5e7eb"
S_FIELD, S_SOFT, S_DARK = "#f3f4f6", "#f9fafb", "#111827"
S_FONT = ("Segoe UI", 10) if IS_WIN else ("Helvetica", 12)
S_SMALL = (S_FONT[0], S_FONT[1] - 1)
S_BOLD = (S_FONT[0], S_FONT[1], "bold")
STRIP_FONT = ("Segoe UI", 11) if IS_WIN else ("Helvetica", 13)
STRIP_NUM_FONT = ("Segoe UI", 9) if IS_WIN else ("Helvetica", 11)
STRIP_RADIUS = 14


def x_length(text):
    """Characters as X counts them: each link is 23, emoji count double."""
    urls = re.findall(r"https?://\S+|\b[\w.-]+\.(?:com|org|io|dev|app)/\S*", text)
    return len(text) + sum(23 - len(u) for u in urls) + sum(1 for c in text if ord(c) > 0xFFFF)


def round_window(hwnd, border_hex=None):
    """Windows 11: smooth rounded corners (and a thin border colour) for one of our windows."""
    import ctypes
    from ctypes import wintypes as W
    try:
        dwm = ctypes.WinDLL("dwmapi")
        dwm.DwmSetWindowAttribute.argtypes = [W.HWND, W.DWORD, ctypes.c_void_p, W.DWORD]
        pref = ctypes.c_int(2)                                     # DWMWCP_ROUND
        ok = dwm.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(pref), 4) == 0
        if ok and border_hex:
            r, g, b = (int(border_hex[i:i + 2], 16) for i in (1, 3, 5))
            color = W.DWORD(r | g << 8 | b << 16)
            dwm.DwmSetWindowAttribute(hwnd, 34, ctypes.byref(color), 4)
        return ok
    except Exception:
        return False


class Session:
    """A small tooltip bubble at the caret/selection. Keeps history so you can ask follow-ups."""

    def __init__(self, app, anchor, below_y, bounds, shot, hwnd=None, proc="", selection="", ctx=None):
        self.app, self.anchor, self.below_y, self.bounds, self.shot = app, anchor, below_y, bounds, shot
        self.hwnd, self.proc, self.selection, self.ctx = hwnd, proc, selection, ctx
        self.msgs = [{"role": "system", "content": SYSTEM + (f"\nThe user's name is {USER_NAME}; use it to "
                      "sign emails or messages you draft for them." if USER_NAME else "")}]
        self.busy = False
        self.alive = True
        self.last_answer = ""
        self.recorder = Recorder()
        self.decision = None
        self.decided = threading.Event()
        self.moved = False
        self.acting = False          # True while we are typing into the target app
        self.away = False            # True while hidden because the user is in another window
        self.had_focus = False
        self.title = window_title(hwnd)

        win = self.win = tk.Toplevel(app.root)
        win.overrideredirect(True)                 # no title bar: a card at the cursor
        win.attributes("-topmost", True)
        win.configure(bg=S_LINE)                   # thin grey edge (rounded on Windows 11)
        win.bind("<Escape>", lambda e: self.escape())
        box = self.box = tk.Frame(win, bg=S_BG, padx=12, pady=10)
        box.pack(padx=1, pady=1)

        # selection chip
        if selection:
            preview = " ".join(selection.split())
            preview = preview[:60] + ("…" if len(preview) > 60 else "")
            tk.Label(box, text=f"“{preview}”", bg=S_BG, fg=S_MUTED, font=S_SMALL, anchor="w",
                     justify="left", wraplength=WRAP).pack(fill="x", pady=(0, 6))
        if not ctx and not selection:
            log("couldn't read text at the caret")   # shown in Settings → Log, not in the card

        # input row
        row = tk.Frame(box, bg=S_BG)
        row.pack(fill="x")
        handle = tk.Label(row, text="✦", bg=S_BG, fg=S_DARK, font=S_BOLD, cursor="fleur")
        handle.pack(side="left", padx=(0, 6))
        handle.bind("<ButtonPress-1>", self.drag_start)
        handle.bind("<B1-Motion>", self.drag_move)
        self.entry = tk.Entry(row, bg=S_FIELD, fg=S_FG, insertbackground=S_FG, relief="flat",
                              font=S_FONT, width=34, highlightthickness=0)
        self.entry.pack(side="left", fill="x", expand=True, ipady=4)
        self.entry.bind("<Return>", lambda e: self.submit())
        self.placeholder = "Ask about the selection…" if selection else "What should I do here?"
        self.set_placeholder()
        self.entry.bind("<FocusIn>", lambda e: self.clear_placeholder())
        self.entry.bind("<Key>", lambda e: self.clear_placeholder())
        self.mic = self.icon(row, "🎤", self.toggle_mic)
        self.icon(row, "✕", self.close)

        # body (hidden until there is something to show)
        self.status = tk.Label(box, bg=S_BG, fg=S_MUTED, font=S_FONT, anchor="w")
        self.answer = tk.Label(box, bg=S_BG, fg=S_FG, font=S_FONT, anchor="w", justify="left", wraplength=WRAP)
        self.steps = tk.Label(box, bg=S_BG, fg=S_MUTED, font=("Consolas", 9) if IS_WIN else ("Menlo", 11),
                              anchor="w", justify="left", wraplength=WRAP)
        self.confirm = tk.Frame(box, bg=S_BG)
        self.confirm_label = tk.Label(self.confirm, bg=S_BG, fg=S_MUTED, font=S_FONT)
        self.confirm_label.pack(side="left")
        self.button(self.confirm, "Stop", lambda: self.decide(False), S_FIELD).pack(side="right")
        self.button(self.confirm, "Run ⏎", lambda: self.decide(True), S_DARK).pack(side="right", padx=4)
        self.summary = tk.Label(box, bg=S_BG, fg=S_MUTED, font=S_FONT, anchor="w", justify="left", wraplength=WRAP)
        self.chips = tk.Frame(box, bg=S_BG)
        self.tools = tk.Frame(box, bg=S_BG)
        self.button(self.tools, "Copy", self.copy_answer, S_FIELD).pack(side="left")
        self.button(self.tools, "Insert", self.insert_answer, S_FIELD).pack(side="left", padx=4)

        self.on_log = None                         # the log lives in Settings → Log now
        self.spinning = False
        self.place()
        if IS_WIN:
            win.after(10, lambda: round_window(self.my_hwnd(), S_LINE))
        win.after(30, self.grab_focus)
        if not has_ai_key() or client is None:
            self.no_key()
        elif AUTO_SUGGEST:
            self.set_status("Reading…")
            threading.Thread(target=self.suggest, daemon=True).start()
        if IS_WIN:
            win.after(400, self.watch)

    # --- only show where it was called
    def my_hwnd(self):
        import ctypes
        return ctypes.windll.user32.GetAncestor(self.win.winfo_id(), 2)   # GA_ROOT

    def watch(self):
        """Close (idle) or hide (working) when the user leaves the window/tab it was opened in."""
        if not self.alive:
            return
        try:
            if not self.acting:
                import ctypes
                fg = ctypes.windll.user32.GetForegroundWindow()
                mine = self.my_hwnd()
                in_target = fg == self.hwnd and window_title(self.hwnd) == self.title
                if fg == mine:
                    self.had_focus = True
                elif not self.busy:
                    # idle: clicking elsewhere, switching app or tab closes it, like a tooltip
                    if self.had_focus or not in_target:
                        self.close()
                        return
                elif not in_target and not self.away:
                    self.away = True             # working: hide while you're elsewhere
                    self.win.withdraw()
                elif in_target and self.away:
                    self.away = False
                    self.win.deiconify()
                    self.win.attributes("-topmost", True)
        except Exception:
            pass
        self.win.after(250, self.watch)

    # --- widgets
    def icon(self, parent, text, cmd):
        lbl = tk.Label(parent, text=text, bg=S_BG, fg=S_MUTED, font=S_FONT, cursor="hand2", padx=4)
        lbl.pack(side="left")
        lbl.bind("<Button-1>", lambda e: cmd())
        return lbl

    def button(self, parent, text, cmd, bg):
        b = tk.Label(parent, text=text, bg=bg, fg="#ffffff" if bg == S_DARK else S_FG, font=S_FONT,
                     padx=10, pady=3, cursor="hand2")
        b.bind("<Button-1>", lambda e: cmd())
        return b

    def set_placeholder(self):
        self.entry.delete(0, "end")
        self.entry.insert(0, self.placeholder)
        self.entry.configure(fg=S_MUTED)
        self.has_placeholder = True

    def clear_placeholder(self):
        if self.has_placeholder:
            self.entry.delete(0, "end")
            self.entry.configure(fg=S_FG)
            self.has_placeholder = False

    def grab_focus(self):
        self.win.lift()
        self.win.focus_force()
        self.entry.focus_set()
        self.entry.icursor("end")

    # --- placement: just under the caret/selection, kept on screen
    def place(self):
        self.win.update_idletasks()
        w, h = self.win.winfo_reqwidth(), self.win.winfo_reqheight()
        left, top, right, bottom = self.bounds
        if self.moved:
            x, y = self.win.winfo_x(), self.win.winfo_y()
        else:
            x = self.anchor[0] - 12
            y = self.below_y + 8
            if y + h > bottom - 8:                       # no room below: go above
                y = self.anchor[1] - h - 24
        x = max(left + 8, min(x, right - w - 8))
        y = max(top + 8, min(y, bottom - h - 8))
        self.win.geometry(f"+{int(x)}+{int(y)}")

    def drag_start(self, e):
        self.drag = (e.x_root - self.win.winfo_x(), e.y_root - self.win.winfo_y())

    def drag_move(self, e):
        self.moved = True
        self.win.geometry(f"+{e.x_root - self.drag[0]}+{e.y_root - self.drag[1]}")

    def move_to(self, anchor, below_y, bounds):
        self.anchor, self.below_y, self.bounds, self.moved = anchor, below_y, bounds, False
        self.place()
        self.grab_focus()

    # --- thread-safe UI updates (agent thread -> UI thread)
    def ui(self, fn):
        def safe():
            if self.alive:
                fn()
        self.app.q.put(safe)

    def show_part(self, widget, text=None, **cfg):
        if text is not None:
            widget.configure(text=text, **cfg)
        if text:
            widget.pack(fill="x", pady=(6, 0))
        else:
            widget.pack_forget()
        self.place()

    def set_status(self, text, color=None):
        def _s():
            self.status_text = text or ""
            self.show_part(self.status, text, fg=color or S_MUTED)
            if self.status_text.endswith("…") and not self.spinning:
                self.spinning = True
                self.spin()
        self.ui(_s)

    def set_answer(self, text):
        self.last_answer = text
        def _a():
            self.show_part(self.answer, text)
            self.tools.pack_forget()
            if text:
                self.tools.pack(anchor="w", pady=(6, 0))
            self.place()
        self.ui(_a)

    def set_steps(self, text, risky=False):
        self.ui(lambda: self.show_part(self.steps, text, fg=RED if risky else S_MUTED))

    # --- loading indicator: any status ending in "…" spins until it changes
    SPIN = "◐◓◑◒"

    def spin(self, i=0):
        text = getattr(self, "status_text", "")
        if not self.alive or not text.endswith("…"):
            self.spinning = False
            return
        self.status.configure(text=f"{self.SPIN[i % 4]}  {text}")
        self.win.after(120, lambda: self.spin(i + 1))

    # --- analyse the spot and offer next actions
    def suggest(self):
        if not self.ctx and not USE_SCREENSHOTS:
            self.set_status("")                 # nothing to analyse; just let the user type
            return
        content = []
        if self.ctx:
            content.append({"type": "text", "text": f"App: {self.proc or 'unknown'}\nTEXT AROUND CARET:\n"
                                                    + format_context(self.ctx)})
        if USE_SCREENSHOTS and self.shot:
            content.append({"type": "image_url", "image_url": {"url": self.shot[1], "detail": "high"}})
        try:
            log("→ suggest: " + " | ".join(c["text"] for c in content if c["type"] == "text"))
            t0 = time.time()
            resp = client.chat.completions.create(model=MODEL, messages=[
                {"role": "system", "content": SUGGEST_SYSTEM}, {"role": "user", "content": content}])
            raw = resp.choices[0].message.content
            log(f"← suggest ({time.time() - t0:.1f}s): {raw}")
            data = parse_reply(raw)
        except Exception as e:
            log("suggest failed:\n" + traceback.format_exc())
            self.set_status(f"Error: {e}", RED)
            return
        if self.busy or len(self.msgs) > 1:
            return                    # the user already asked something
        self.set_status("")
        self.ui(lambda: self.show_suggestions(data.get("summary", ""), data.get("suggestions") or []))

    def show_suggestions(self, summary, suggestions):
        self.show_part(self.summary, summary)
        for child in self.chips.winfo_children():
            child.destroy()
        for text in suggestions[:4]:
            chip = self.button(self.chips, text, lambda t=text: self.ask(t), S_FIELD)
            chip.pack(side="left", padx=(0, 4), pady=2)
        if suggestions:
            self.chips.pack(anchor="w", pady=(6, 0))
        self.place()

    def hide_suggestions(self):
        self.summary.pack_forget()
        self.chips.pack_forget()
        self.place()

    def ask(self, text):
        self.clear_placeholder()
        self.entry.delete(0, "end")
        self.entry.insert(0, text)
        self.submit()

    # --- input
    def toggle_mic(self):
        if self.recorder.stream is None:
            if not has_ai_key() or client is None:
                self.no_key()                        # speech-to-text also uses the AI key
                return
            self.recorder.start()
            self.mic.configure(text="■", fg=RED)
            self.set_status("Listening… click ■ to stop")
        else:
            self.mic.configure(text="…", fg=S_MUTED)

            def work():
                try:
                    text = self.recorder.stop_and_transcribe()
                except Exception as e:
                    self.set_status(f"Mic error: {e}", RED)
                    text = ""
                self.ui(lambda: self.mic.configure(text="🎤"))
                if text.strip():
                    self.ui(lambda: (self.clear_placeholder(), self.entry.delete(0, "end"),
                                     self.entry.insert(0, text), self.submit()))
                else:
                    self.set_status("")
            threading.Thread(target=work, daemon=True).start()

    def no_key(self):
        """No AI key yet: say so in the card, with a way to add one (typing help works without it)."""
        self.set_status("The assistant needs an AI key. Word suggestions and auto-correct work without one.")
        def add_button():
            if not self.alive:
                return
            row = tk.Frame(self.box, bg=S_BG)
            row.pack(fill="x", pady=(6, 0))
            self.button(row, "Add a key in Settings", lambda: (self.close(), self.app.q.put(
                lambda: (self.app.open_settings(), self.app.settings_win.show_tab("Settings")))), S_DARK).pack(side="left")
            self.place()
        self.ui(add_button)

    def submit(self):
        goal = "" if self.has_placeholder else self.entry.get().strip()
        if not goal or self.busy:
            return
        if not has_ai_key() or client is None:
            self.no_key()
            return
        self.entry.delete(0, "end")
        self.busy = True
        self.hide_suggestions()
        self.set_answer("")
        self.set_steps("")
        threading.Thread(target=self.agent, args=(goal,), daemon=True).start()

    def copy_answer(self):
        pyperclip.copy(self.last_answer)
        self.set_status("Copied ✓")

    def insert_answer(self):
        if self.busy or not self.last_answer:
            return
        text = self.last_answer

        def work():
            self.hide()
            run_actions([{"type": "type", "text": text}], self.anchor,
                        click_to_focus=FOCUS_CLICK and not self.hwnd and not self.selection)
            self.selection = ""
            self.close_from_thread()
        threading.Thread(target=work, daemon=True).start()

    # --- confirmation
    def ask_confirm(self, risky):
        self.decided.clear()

        def show():
            self.confirm_label.configure(text="⚠ Risky. Run?" if risky else "Run these?",
                                         fg=RED if risky else S_MUTED)
            self.confirm.pack(fill="x", pady=(6, 0))
            self.place()
            self.win.bind("<Return>", lambda e: self.decide(True))
            self.grab_focus()
        self.ui(show)
        self.decided.wait()
        return self.decision

    def decide(self, ok):
        self.decision = ok
        self.confirm.pack_forget()
        self.win.unbind("<Return>")
        self.place()
        self.decided.set()

    def escape(self):
        if self.confirm.winfo_ismapped():
            self.decide(False)
        else:
            self.close()

    # --- the loop: look -> think -> confirm -> act -> look again
    def agent(self, goal):
        try:
            if len(self.msgs) > 1:      # follow-up: take a fresh look first
                self.hide()
                self.look()
                self.show()
            for step in range(MAX_STEPS):
                strip_old_images(self.msgs)
                if step == 0:
                    text = f"Goal: {goal}"
                    if self.proc:
                        text += f"\nActive app: {self.proc}"
                    if self.ctx:
                        text += "\nTEXT AROUND CARET (this section only):\n" + format_context(self.ctx)
                    if self.selection:
                        text += f'\nSELECTED TEXT:\n"""\n{self.selection}\n"""'
                else:
                    text = "Your actions ran. TEXT AROUND CARET now:\n" + (
                        format_context(self.ctx) if self.ctx else "(could not read)") + \
                        "\nContinue toward the goal, or set done true."
                content = [{"type": "text", "text": text}]
                if USE_SCREENSHOTS and self.shot:
                    content += [{"type": "image_url", "image_url": {"url": u, "detail": "high"}}
                                for u in self.shot]
                self.msgs.append({"role": "user", "content": content})
                self.set_status("Thinking…")
                log(f"→ step {step + 1}: {text}")
                t0 = time.time()
                resp = client.chat.completions.create(model=MODEL, messages=self.msgs)
                raw = resp.choices[0].message.content
                log(f"← step {step + 1} ({time.time() - t0:.1f}s): {raw}")
                self.msgs.append({"role": "assistant", "content": raw})
                reply = parse_reply(raw)
                self.set_status("")

                actions = reply.get("actions") or []
                if not actions:
                    self.set_steps("")
                    self.set_answer(reply.get("say", ""))
                    break
                risky = bool(reply.get("risky"))
                self.ui(lambda r=reply: self.show_part(self.answer, r.get("say", "")))
                self.set_steps("\n".join("› " + describe(a) for a in actions), risky)
                if not self.ask_confirm(risky):
                    self.set_steps("")
                    self.set_status("Stopped.")
                    break

                self.set_steps("")
                self.hide()
                # Windows: refocus the original window (keeps the selection). Elsewhere: click the spot,
                # unless there is a selection a click would destroy.
                log("running: " + "; ".join(describe(a) for a in actions))
                run_actions(actions, self.anchor,
                            click_to_focus=FOCUS_CLICK and not self.hwnd and not self.selection)
                self.selection = ""      # it has been replaced/used
                self.look()
                self.show()
                if reply.get("done"):
                    self.set_status("Done ✓")
                    break
            else:
                self.set_status(f"Stopped after {MAX_STEPS} steps.")
        except pyautogui.FailSafeException:
            self.show()
            self.set_status("Aborted (mouse moved to a corner).", RED)
        except Exception as e:
            log("agent failed:\n" + traceback.format_exc())
            self.show()
            self.set_status(f"Error: {e}", RED)
        finally:
            self.busy = False

    def look(self):
        """Re-read the caret (and screenshot if enabled) while the target app has focus."""
        try:
            self.ctx = read_caret_context(self.proc)
        except Exception:
            self.ctx = None
            log("look failed:\n" + traceback.format_exc())
        if USE_SCREENSHOTS:
            self.shot = capture(self.anchor)

    def hide(self):
        self.acting = True
        done = threading.Event()

        def _hide():
            if self.hwnd:            # give focus back while we still own the foreground
                try:
                    focus_window(self.hwnd)
                except Exception:
                    pass
            self.win.withdraw()
            done.set()
        self.ui(_hide)
        done.wait(2)
        time.sleep(0.25)

    def show(self):
        def _show():
            self.title = window_title(self.hwnd)   # commands can change the window title
            self.away = False
            self.had_focus = False
            self.win.deiconify()
            self.win.attributes("-topmost", True)
            self.grab_focus()
            self.acting = False
        self.ui(_show)

    def close_from_thread(self):
        self.app.q.put(self.close)

    def close(self):
        if not self.alive:
            return
        self.alive = False
        if self.on_log and self.on_log in LOG_LISTENERS:
            LOG_LISTENERS.remove(self.on_log)
        if self.recorder.stream is not None:
            self.recorder.stream.stop()
            self.recorder.stream.close()
        self.decision = False
        self.decided.set()
        self.win.destroy()
        if self.app.session is self:
            self.app.session = None


class WebCaret:
    """Listens on 127.0.0.1 for the Careter browser extension (chrome-extension/), which reports the
    text before the caret and where the caret is on the page. Only browser extensions can post here
    (web pages can't fake an extension's Origin), and only this PC can reach it."""

    def __init__(self, port=WEB_PORT):
        import http.server
        self.latest, self.lock = None, threading.Lock()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                origin = self.headers.get("Origin", "")
                size = int(self.headers.get("Content-Length") or 0)
                if self.path != "/caret" or not origin.startswith("chrome-extension://") or size > 65536:
                    self.send_response(403)
                    self.end_headers()
                    return
                try:
                    data = json.loads(self.rfile.read(size).decode("utf-8"))
                    if isinstance(data, dict):
                        data["at"] = time.time()
                        with owner.lock:
                            owner.latest = data
                    self.send_response(204)
                except ValueError:
                    self.send_response(400)
                self.end_headers()

            def do_GET(self):                      # "is Careter running?" for the browser extension
                self.send_response(204 if self.path == "/ping" else 404)
                self.end_headers()

            def log_message(self, *args):          # quiet: never log what was typed
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def fresh(self, since, max_age=1.5):
        """The latest report if it came after `since` (your last keystroke) and is recent."""
        with self.lock:
            d = self.latest
        if d and d["at"] >= since - 0.05 and time.time() - d["at"] <= max_age:
            return d
        return None

    def recent_password(self, max_age=5.0):
        with self.lock:
            d = self.latest
        return bool(d and d.get("password") and time.time() - d["at"] <= max_age)


def browser_viewport(hwnd):
    """Screen (left, top) of the web page area in a Chromium browser window: its largest visible
    'Chrome_RenderWidgetHostHWND' child. None if not found."""
    import ctypes
    from ctypes import wintypes as W
    u = ctypes.WinDLL("user32")
    found = []
    PROC = ctypes.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)

    def visit(child, _):
        name = ctypes.create_unicode_buffer(64)
        u.GetClassNameW(child, name, 64)
        if name.value == "Chrome_RenderWidgetHostHWND" and u.IsWindowVisible(child):
            r = W.RECT()
            if u.GetWindowRect(child, ctypes.byref(r)):
                found.append(((r.right - r.left) * (r.bottom - r.top), r.left, r.top))
        return True
    u.EnumChildWindows.argtypes = [W.HWND, PROC, W.LPARAM]
    u.GetClassNameW.argtypes = [W.HWND, W.LPWSTR, ctypes.c_int]
    u.IsWindowVisible.argtypes = [W.HWND]
    u.GetWindowRect.argtypes = [W.HWND, ctypes.POINTER(W.RECT)]
    u.EnumChildWindows(hwnd, PROC(visit), 0)
    if not found:
        return None
    _, left, top = max(found)
    return left, top


def web_point(web, hwnd):
    """Screen point (x, top, bottom) of the caret from an extension report: page area origin +
    CSS pixels x devicePixelRatio (which already includes zoom and Windows scaling)."""
    rect = web.get("rect")
    if not rect:
        return None
    vp = browser_viewport(hwnd)
    if not vp:
        return None
    dpr = float(web.get("dpr") or 1)
    return (int(vp[0] + rect["left"] * dpr), int(vp[1] + rect["top"] * dpr),
            int(vp[1] + rect["bottom"] * dpr))


class Predictions:
    """Next-word suggestions while you type, from a small local model (predict.py).

    A keyboard hook keeps a short buffer of what you typed in the current window. After a short
    pause it reads the text before the caret (UI Automation, or the buffer if the app hides its
    text), asks the model for the next words and shows them in a strip at the caret that never
    takes focus. Keys 1-5 pick a suggestion (Tab = the first). Those keys are only
    captured while the strip is showing. Typed text stays in memory and is never logged."""

    BUF_MAX = 400

    def __init__(self, app):
        import ctypes
        self.user32 = ctypes.windll.user32
        self.app = app
        self.enabled = PREDICT
        self.predictor = None
        self.buf, self.buf_hwnd, self.buf_proc = "", None, ""
        self.speller, self.last_fix, self.fixing = None, None, False
        self.fast_predictor = None
        self.sel, self.rows, self.shown_gen, self.ghost = 0, [], -1, None
        self.ghost_point, self.ghost_bg = None, (None, None)
        self.uia = UiaReader()
        self.epoch, self.last_exact = 0, None       # last exact caret spot in this window (for pass 1)
        self.last_seen = None                       # (window, epoch, text) read while you typed
        self.snaps = {}                             # window -> (keys typed there before a click, when)
        self.keys_used = ("", None)                 # the typed-keys text the last prediction used
        self.user_model = None
        self.last_key_at = 0.0
        self.web = None
        try:
            self.web = WebCaret()
        except OSError:
            log(f"predict: browser helper port {WEB_PORT} is busy; browsers fall back to keys")
        self.shown, self.visible = [], False
        self.gen = 0                       # bumps on every keystroke; stale predictions are dropped
        self.timer = None
        self.swallow_up = set()            # keys whose key-up we also hide from the app
        self.strip = None
        self.strip_rect = (0, 0, 0, 0)
        self.w = self.typed_user32()
        app.q.put(self.build_strip)
        threading.Thread(target=self.load, daemon=True).start()
        self.kb = keyboard.Listener(on_press=self.on_press, win32_event_filter=self.filter)
        self.kb.start()
        self.ms = mouse.Listener(on_click=self.on_click)
        self.ms.start()

    @staticmethod
    def typed_user32():
        """Our own user32 with exact argument types. Window handles are 64-bit pointers: passed as
        plain ints they can get cut short and the call silently fails (the strip never shows)."""
        import ctypes
        from ctypes import wintypes as W
        u = ctypes.WinDLL("user32")
        sig = {
            "GetAncestor": ([W.HWND, W.UINT], W.HWND),
            "GetWindowLongW": ([W.HWND, ctypes.c_int], ctypes.c_long),
            "SetWindowLongW": ([W.HWND, ctypes.c_int, ctypes.c_long], ctypes.c_long),
            "ShowWindow": ([W.HWND, ctypes.c_int], W.BOOL),
            "EnableWindow": ([W.HWND, W.BOOL], W.BOOL),
            "IsWindowVisible": ([W.HWND], W.BOOL),
            "GetForegroundWindow": ([], W.HWND),
            "SetForegroundWindow": ([W.HWND], W.BOOL),
            "SetWindowPos": ([W.HWND, W.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              W.UINT], W.BOOL),
        }
        for name, (args, res) in sig.items():
            fn = getattr(u, name)
            fn.argtypes, fn.restype = args, res
        return u

    # --- model
    def load(self):
        """Word lists first (ready in seconds), then the LLM on top when the engine is "llm"."""
        if not self.enabled or getattr(self, "loading", False):
            return
        self.loading = True
        try:
            t0 = time.time()
            from predict import HybridPredictor, NGramPredictor, PredictorProcess
            log("predict: loading word lists…")
            folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngrams")
            ngram = NGramPredictor(folder, log=log)
            self.predictor = self.fast_predictor = ngram
            personal = lambda engine: engine
            if LEARN:
                from predict import PersonalPredictor, UserModel
                self.user_model = UserModel(os.path.join(folder, "user_model.dat"),
                                            legacy=os.path.join(folder, "user_model.json"))
                personal = lambda engine: PersonalPredictor(engine, self.user_model)
                self.predictor = self.fast_predictor = personal(ngram)
                threading.Thread(target=self.autosave, daemon=True).start()
                log(f"predict: learning on, knows {len(self.user_model.uni)} of your words")
            if AUTOCORRECT:
                from predict import Speller
                self.speller = Speller(ngram, folder)
                self.speller.user_model = self.user_model
            log(f"predict: word lists ready in {time.time() - t0:.1f}s")
            if PREDICT_ENGINE != "llm":
                return
            t0 = time.time()
            log(f"predict: loading {PREDICT_MODEL} (first run downloads it)…")
            try:
                llm = PredictorProcess(PREDICT_MODEL)              # model runs in its own process
            except Exception as e:
                if "No module named" in str(e):
                    log("predict: LLM needs torch + transformers: pip install -r requirements-llm.txt. "
                        "Using word lists.")
                else:
                    log("predict: LLM failed to load, using word lists:\n" + traceback.format_exc())
                return
            hybrid = HybridPredictor(llm, ngram, PREDICT_LLM_TIMEOUT)
            warm = llm.predict("Hello, how are ", timeout=30)    # warm-up
            self.predictor = personal(hybrid)
            log(f"predict: LLM ready in {time.time() - t0:.1f}s, test: {[w for w, _ in warm]}")
        except Exception:
            log("predict: failed to load:\n" + traceback.format_exc())
        finally:
            self.loading = False

    def autosave(self):
        import atexit
        atexit.register(self.save_learned)
        while True:
            time.sleep(30)
            self.save_learned()

    def save_learned(self):
        try:
            if self.user_model:
                self.user_model.save()
        except Exception:
            log("predict: could not save what it learned:\n" + traceback.format_exc())

    def learn_last(self):
        """A word was just finished (space/enter/accepted): count it, and the word before it."""
        if not LEARN or not self.user_model or self.buf_proc in TERMINALS or self.in_browser_password():
            return
        m = re.search(r"((?:[^\W\d_]|['’\-])+)\s\Z", self.buf)
        if not m:
            return
        word, before = m.group(1), self.buf[: m.start()]
        if not before.strip() or not before[-1].isspace():
            return                             # first word in a box (maybe a username/password)
        tail = before.rstrip()
        sentence_start = tail[-1] in ".!?\n"
        if word[:1].isupper() and not sentence_start and word != "I" and not word.startswith(("I'", "I’")):
            return                             # a capitalised word mid-sentence: a name or place
        pm = re.search(r"((?:[^\W\d_]|['’\-])+)\Z", tail)
        prev = pm.group(1).lower() if (pm and not sentence_start) else None
        self.user_model.learn(prev, word.lower().replace("’", "'"))

    def toggle(self):
        self.enabled = not self.enabled
        self.reset()
        log(f"predict: {'on' if self.enabled else 'off'}")
        if self.enabled and self.predictor is None:
            threading.Thread(target=self.load, daemon=True).start()
        x, y = pyautogui.position()
        self.app.q.put(lambda: self.flash(f"Word prediction {'on' if self.enabled else 'off'}", (x, y, y + 16)))

    # --- keyboard
    def mods(self):
        down = lambda vk: bool(self.user32.GetAsyncKeyState(vk) & 0x8000)
        return down(0x11), down(0x12), down(0x10), down(0x5B) or down(0x5C)   # ctrl, alt, shift, win

    def filter(self, msg, data):
        """Runs inside the low-level hook (must be quick). Returns False to skip on_press."""
        if data.flags & 0x10:                  # LLKHF_INJECTED: keys we (or other tools) typed
            return False
        vk, down = data.vkCode, msg in (0x100, 0x104)          # WM_KEYDOWN, WM_SYSKEYDOWN
        if not down and vk in self.swallow_up:
            self.swallow_up.discard(vk)
            self.kb.suppress_event()
        if down and vk in (0x20, 0x08) and self.enabled and AUTOCORRECT and self.speller and not self.fixing:
            ctrl, alt, _, win = self.mods()
            if not (ctrl or alt or win):
                job = self.check_fix() if vk == 0x20 else self.check_undo()
                if job:
                    self.fixing = True
                    self.swallow_up.add(vk)
                    threading.Thread(target=job, daemon=True).start()
                    self.kb.suppress_event()   # we type the space / undo ourselves
        if down and self.visible and self.shown:
            ctrl, alt, shift, win = self.mods()
            pick = None
            plain = not (ctrl or alt or shift or win)
            if plain and vk in (0x26, 0x28):          # up / down: move the highlight
                self.swallow_up.add(vk)
                step = -1 if vk == 0x26 else 1
                self.app.q.put(lambda: self.move(step))
                self.kb.suppress_event()
            if (vk == 0x09 or (vk == 0x0D and ENTER_PICKS)) and plain:   # Tab / Enter: the highlighted one
                pick = min(self.sel, len(self.shown) - 1)
            elif plain:
                n = len(self.shown)
                if 0x31 <= vk < 0x31 + n:              # 1..5 on the number row
                    pick = vk - 0x31
                elif 0x61 <= vk < 0x61 + n:            # 1..5 on the numpad
                    pick = vk - 0x61
            if pick is not None:
                choice = self.shown[pick]
                self.swallow_up.add(vk)
                threading.Thread(target=self.accept, args=(choice,), daemon=True).start()
                self.kb.suppress_event()       # the app never sees this key
        return True

    def on_press(self, key):
        if not self.enabled:
            return
        if not getattr(self, "seen_key", False):
            self.seen_key = True
            log("predict: keyboard hook is receiving keys")
        self.last_key_at = time.time()
        hwnd = self.user32.GetForegroundWindow()
        if hwnd != self.buf_hwnd:              # another window: what we typed before doesn't apply
            self.snapshot()
            self.buf, self.buf_hwnd, self.last_fix = "", hwnd, None
            self.buf_proc = foreground_window()[1]
        K = keyboard.Key
        if key in (K.shift, K.shift_r, K.caps_lock, K.ctrl, K.ctrl_l, K.ctrl_r,
                   K.alt, K.alt_l, K.alt_r, K.alt_gr, K.cmd, K.cmd_r):
            return                             # a modifier on its own changes nothing
        ctrl, alt, _, win = self.mods()
        char = getattr(key, "char", None)
        if key == K.backspace and not (ctrl or alt):
            self.buf = self.buf[:-1]
        elif key == K.space and not (ctrl or alt or win):
            self.buf += " "
            self.learn_last()
            self.buf = self.buf[-self.BUF_MAX:]
            self.schedule(0.02)
            return
        elif key == K.enter and not (ctrl or alt):
            self.buf += "\n"
            self.learn_last()
        elif char and char.isprintable() and not win and (ctrl == alt):   # ctrl+alt = AltGr chars
            self.buf += char
        else:                                  # arrows, Home, Esc, shortcuts (Ctrl+V…): caret/text unknown
            self.reset()
            return
        self.buf = self.buf[-self.BUF_MAX:]
        self.schedule()

    # --- auto-correct (runs inside the keyboard hook: only fast dictionary lookups here)
    def check_fix(self):
        """Space pressed: if the word just typed is a clear typo, return a job that replaces it."""
        if self.buf_proc in TERMINALS or self.in_browser_password():
            return None
        m = re.search(r"([^\W\d_]+)\Z", self.buf)
        if not m:
            return None
        word, before = m.group(1), self.buf[: m.start()]
        if not before.strip():
            # The typed-keys buffer starts at this word (e.g. you clicked into existing text). If the
            # text read from the app while you typed it shows words before it, it isn't the first word
            # in the box, so it may be corrected.
            ls = getattr(self, "last_seen", None)
            seen = ls[2] if (ls and ls[0] == self.buf_hwnd and ls[1] == self.epoch) else ""
            typed = re.search(r"([^\W\d_]*)\Z", seen).group(1)     # the part of this word typed then
            context = seen[: len(seen) - len(typed)] if (seen and word.startswith(typed)) else ""
            if context.strip() and context[-1].isspace():
                before = context
            else:
                self.fix_why("first word in a box (could be a username or password)")
                return None
        # Never the first thing typed in a field (protects passwords and usernames), and never
        # glued to symbols like "x.teh" or "@teh".
        if not before[-1].isspace():
            return None
        tail = before.rstrip()
        pm = re.search(r"([^\W\d_]+)\Z", tail)
        prev = pm.group(1).lower() if (pm and tail[-1] not in ".!?") else None
        if not word.islower():
            self.fix_why("capitalised word (names and acronyms are left alone)")
            return None
        new = self.speller.correct(word, prev)
        if not new or new == word:
            return None
        return lambda: self.apply_fix(word, new)

    def fix_why(self, reason):
        """Log why a word wasn't corrected, once per change (never the word itself)."""
        if reason != getattr(self, "last_fix_why", None):
            self.last_fix_why = reason
            log(f"predict: auto-correct skipped: {reason}")

    def in_browser_password(self):
        web = getattr(self, "web", None)
        return bool(web and self.buf_proc in BROWSERS and web.recent_password())

    def apply_fix(self, word, new):
        try:
            self.hide()
            pyautogui.press("backspace", presses=len(word), _pause=False)
            pyautogui.write(new + " ", _pause=False)
            self.buf = (self.buf[: -len(word)] + new + " ")[-self.BUF_MAX:]
            self.last_fix = (word, new, self.buf)
            self.learn_last()
            log(f"predict: auto-corrected a {len(word)}-letter word")     # never the word itself
        except Exception:
            log("predict: auto-correct failed:\n" + traceback.format_exc())
        finally:
            self.fixing = False
        self.schedule()

    def check_undo(self):
        """Backspace right after a correction: put back what you typed and remember it."""
        if not self.last_fix or self.buf != self.last_fix[2]:
            self.last_fix = None
            return None
        word, new, _ = self.last_fix
        self.last_fix = None
        return lambda: self.apply_undo(word, new)

    def apply_undo(self, word, new):
        try:
            self.hide()
            pyautogui.press("backspace", presses=len(new) + 1, _pause=False)
            pyautogui.write(word, _pause=False)
            self.buf = (self.buf[: -(len(new) + 1)] + word)[-self.BUF_MAX:]
            self.speller.remember(word)
            log("predict: correction undone; that word won't be corrected again")
        except Exception:
            log("predict: undo failed:\n" + traceback.format_exc())
        finally:
            self.fixing = False

    def on_click(self, x, y, button, pressed):
        if not pressed:
            return
        l, t, r, b = self.strip_rect
        if self.visible and l - 8 <= x < r + 8 and t - 8 <= y < b + 8:
            return                             # a click on (or right next to) our own list
        self.reset()                           # the caret probably moved

    def snapshot(self):
        """Remember what was typed in this window (e.g. "know") before a click or a window switch,
        so typing on there ("le") can still be completed ("knowledge") in apps that hide their text."""
        if not hasattr(self, "snaps"):
            self.snaps, self.keys_used = {}, ("", None)
        if self.buf_hwnd and self.buf:
            used, hwnd = self.keys_used
            text = used if (hwnd == self.buf_hwnd and used.endswith(self.buf)) else self.buf
            self.snaps[self.buf_hwnd] = (text[-self.BUF_MAX:], time.time())

    def keys_text(self, hwnd):
        """What we know was typed here: the keys since the last click, or, when that's only part of
        a word, the keys from before the click + these (if that still makes a word)."""
        buf = self.buf
        snap = getattr(self, "snaps", {}).get(hwnd)
        if (snap and buf and not any(c.isspace() for c in buf) and time.time() - snap[1] < 300
                and self.fast_predictor):
            joined = snap[0] + buf
            if self.fast_predictor.predict(joined, 1):
                self.keys_used = (joined, hwnd)
                return joined
        self.keys_used = (buf, hwnd)
        return buf

    def reset(self):
        self.snapshot()
        self.buf, self.last_fix = "", None
        self.epoch += 1                        # the caret may have moved: forget its last position
        self.gen += 1
        if self.timer:
            self.timer.cancel()
        self.hide()

    def schedule(self, delay=None):
        """Predict after a pause: short after a space or a picked word (the next word is wanted now),
        a little longer mid-word so the list doesn't flicker while you type fast."""
        self.gen += 1
        self.hide()
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(PREDICT_DELAY if delay is None else delay, self.run, args=(self.gen,))
        self.timer.daemon = True
        self.timer.start()

    # --- predict
    def run(self, gen):
        """Two passes: an instant one (typed keys or the browser helper + word lists, at the Win32 caret)
        so something shows within ~0.1 s, then the full one (UI Automation text + LLM) that replaces it
        if it's different. Never logs what you typed: only why a suggestion was or wasn't shown."""
        why = lambda msg: DEBUG and log("predict: " + msg)
        try:
            if gen != self.gen or not self.enabled:
                return
            if not self.predictor:
                return why("model not loaded yet")
            hwnd, proc = foreground_window()
            if self.is_ours(hwnd):
                return                         # typing in Careter's own bubble
            if proc in TERMINALS and not PREDICT_IN_TERMINALS:
                return why(f"skipped, {proc} is a terminal")
            if self.app.session and self.app.session.acting:
                return
            from predict import worth_predicting
            web = self.web.fresh(self.last_key_at) if (self.web and proc in BROWSERS) else None
            if web and web.get("password"):
                return why("skipped, password box (browser)")

            pushed = []

            def push(words, point, exact):
                """Queue a list for the screen unless it's what we already showed."""
                if not words or gen != self.gen or (pushed and pushed[-1] == (words, point, exact)):
                    return
                pushed.append((words, point, exact))
                self.app.q.put(lambda w=words, pt=point, ex=exact: self.show(w, pt, gen, exact=ex))

            def remember(point):
                self.last_exact = (hwnd, self.epoch, point)

            def recent():
                """Where the caret last was in this window, if nothing moved it since (approximate)."""
                le = self.last_exact
                return le[2] if le and le[0] == hwnd and le[1] == self.epoch else None

            # pass 1: instant (typed keys / browser helper + word lists)
            fast_text = (web.get("text") or "") if web else self.keys_text(hwnd)
            point1 = web_point(web, hwnd) if web else caret_rect()
            exact1 = bool(point1)
            if exact1:
                remember(point1)
            point1 = point1 or recent()
            if self.fast_predictor and worth_predicting(fast_text) and point1:
                push(self.fast_predictor.predict(fast_text, PREDICT_COUNT), point1, exact1)

            # pass 2: the real text (UI Automation) + word lists, shown at once
            uia = None
            if web:
                source, text = "web", web.get("text") or ""
            else:
                uia = self.uia.read(300, 0.25)
                if uia and uia[2]:
                    self.app.q.put(lambda: gen == self.gen and self.hide())
                    return why("skipped, password box")
                source = "uia" if uia and uia[0].strip() else "keys"
                text = uia[0] if source == "uia" else self.keys_text(hwnd)
            if gen != self.gen:
                return
            if source in ("uia", "web") and text.strip():
                self.last_seen = (hwnd, self.epoch, text)  # lets auto-correct see words before a click
            if not text.strip():
                return why(f"no text before caret in {proc} (uia={'yes' if uia else 'no'})")
            if not worth_predicting(text):
                return                         # start of a sentence: wait for the first letter
            where, point = "uia", (uia[1] if uia else None)
            if web:
                point, where = (point1 if exact1 else None), "web"
            if not point:
                point, where = caret_rect(), "caret"
            exact = bool(point)
            if exact:
                remember(point)
            else:
                point, where = recent(), "last caret spot"
            if not point:
                mx, my = pyautogui.position()
                point, where = (mx, my, my + 16), "mouse"
            if self.fast_predictor and (source != "web" or not pushed):
                push(self.fast_predictor.predict(text, PREDICT_COUNT), point, exact)

            # pass 3: the AI model, replacing the list if it found something different
            if self.predictor is self.fast_predictor:
                return why(f"{len(pushed[-1][0]) if pushed else 0} words by ngram from {source} in {proc}, at {where}")
            t0 = time.time()
            words = self.predictor.predict(text, PREDICT_COUNT, cancelled=lambda: gen != self.gen)
            ms = (time.time() - t0) * 1000
            if gen != self.gen:
                return
            engine = getattr(self.predictor, "last", "ngram")
            why(f"{len(words)} words by {engine} in {ms:.0f}ms from {source} in {proc}, at {where}")
            push(words, point, exact)
        except Exception:
            log("predict failed:\n" + traceback.format_exc())

    def is_ours(self, hwnd):
        from ctypes import wintypes, byref
        pid = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(hwnd, byref(pid))
        return pid.value == os.getpid()

    def accept(self, choice):
        word, insert = choice
        fix = len(insert) - len(insert.lstrip("\b"))     # a fix replaces the typed letters first
        insert = insert[fix:]
        self.gen += 1
        self.hide()
        time.sleep(0.03)
        try:
            if fix:
                pyautogui.press("backspace", presses=fix, _pause=False)
            if insert.isascii():
                pyautogui.write(insert, interval=0.004)
            else:
                paste(insert)
        except Exception:
            log("predict: insert failed:\n" + traceback.format_exc())
            return
        self.buf = ((self.buf[:-fix] if fix else self.buf) + insert)[-self.BUF_MAX:]
        self.learn_last()
        self.schedule(0)                       # offer the word after it straight away

    # --- the strip (UI thread only)
    def build_strip(self):
        try:
            self._build_strip()
            log("predict: strip window ready")
        except Exception:
            self.strip = None
            log("predict: strip window failed:\n" + traceback.format_exc())

    def _build_strip(self):
        w = self.w
        win = self.strip = tk.Toplevel(self.app.root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.configure(bg=STRIP_BG)
        self.strip_box = tk.Frame(win, bg=STRIP_BG, padx=6, pady=6)
        self.strip_box.pack()
        win.geometry("+-2000+-2000")
        win.update_idletasks()
        self.strip_hwnd = w.GetAncestor(win.winfo_id(), 2)          # GA_ROOT
        GWL_EXSTYLE, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW = -20, 0x08000000, 0x80
        style = w.GetWindowLongW(self.strip_hwnd, GWL_EXSTYLE)
        w.SetWindowLongW(self.strip_hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
        # WS_EX_NOACTIVATE above keeps focus in your app even when you click or drag the list
        w.ShowWindow(self.strip_hwnd, 0)            # SW_HIDE
        self.strip_checked = False
        self.round_native = self.round_corners()
        self.build_ghost()

    def build_ghost(self):
        """The grey completion right after the cursor: a borderless, click-through window whose
        background colour is made transparent, so only the grey letters show."""
        g = self.ghost = tk.Toplevel(self.app.root)
        g.overrideredirect(True)
        g.attributes("-topmost", True)
        self.ghost_key = "#010203"
        g.configure(bg=self.ghost_key)
        g.attributes("-transparentcolor", self.ghost_key)
        self.ghost_label = tk.Label(g, text="", bg=self.ghost_key, fg=GHOST_FG, bd=0, padx=0, pady=0)
        self.ghost_label.pack()
        g.geometry("+-2000+-2000")
        g.update_idletasks()
        self.ghost_hwnd = self.w.GetAncestor(g.winfo_id(), 2)
        WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TRANSPARENT = 0x08000000, 0x80, 0x20
        style = self.w.GetWindowLongW(self.ghost_hwnd, -20)
        self.w.SetWindowLongW(self.ghost_hwnd, -20, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TRANSPARENT)
        try:
            g.attributes("-disabled", True)
        except tk.TclError:
            pass
        self.w.ShowWindow(self.ghost_hwnd, 0)

    def line_background(self, x, top, bottom):
        """Colour of the empty space right after the cursor, or None if there's text there (then a
        grey completion would overlap it, so we don't show one)."""
        import ctypes
        u, g = ctypes.WinDLL("user32"), ctypes.WinDLL("gdi32")
        u.GetDC.restype = ctypes.c_void_p
        u.GetDC.argtypes = [ctypes.c_void_p]
        u.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        g.GetPixel.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        g.GetPixel.restype = ctypes.c_uint32
        dc = u.GetDC(None)
        try:
            ys = (top + (bottom - top) // 3, top + (bottom - top) // 2, top + 2 * (bottom - top) // 3)
            colors = {g.GetPixel(dc, int(x + dx), int(y)) for dx in (4, 10, 18, 30, 45) for y in ys}
        finally:
            u.ReleaseDC(None, dc)
        if len(colors) != 1:
            return None
        c = colors.pop()
        if c == 0xFFFFFFFF:                            # CLR_INVALID: off screen
            return None
        return f"#{c & 0xFF:02x}{(c >> 8) & 0xFF:02x}{(c >> 16) & 0xFF:02x}"

    def ghost_why(self, reason):
        """Log why the grey text is or isn't shown, once per change (never the text itself)."""
        if reason != getattr(self, "last_ghost_why", None):
            self.last_ghost_why = reason
            log(f"predict: grey text {'shown' if reason == 'shown' else 'skipped: ' + reason}")

    def place_ghost(self, point):
        if not self.ghost or not self.shown:
            return
        text = self.shown[self.sel][1].rstrip(" ")
        if text.startswith("\b"):                     # a fix replaces the word: nothing to add after it
            self.w.ShowWindow(self.ghost_hwnd, 0)
            return
        x, top, bottom = point
        if self.ghost_bg[0] == point:                  # same spot (arrow keys, better words arrived):
            bg = self.ghost_bg[1]                      # don't re-read the screen, our grey text is on it
        else:
            self.w.ShowWindow(self.ghost_hwnd, 0)
            bg = self.line_background(x, top, bottom)
            self.ghost_bg = (point, bg)
        bg = bg if text else None
        if not bg:
            self.w.ShowWindow(self.ghost_hwnd, 0)
            self.ghost_why("text (or a pattern) right after the cursor, so it would cover it")
            return
        self.ghost_why("shown")
        # the transparent colour = the line's own background, so the grey text's smoothed edges
        # blend into it and look like part of the app
        self.ghost.configure(bg=bg)
        self.ghost.attributes("-transparentcolor", bg)
        px = max(10, min(48, round((bottom - top) * 0.72)))
        self.ghost_label.configure(text=text, bg=bg, font=(STRIP_FONT[0], -px))
        self.ghost.update_idletasks()
        h = self.ghost.winfo_reqheight()
        self.ghost.geometry(f"+{int(x) + 1}+{int(top + ((bottom - top) - h) / 2)}")
        self.ghost.update_idletasks()
        self.w.ShowWindow(self.ghost_hwnd, 4)          # SW_SHOWNOACTIVATE
        self.w.SetWindowPos(self.ghost_hwnd, -1, 0, 0, 0, 0, 0x1 | 0x2 | 0x10)

    def round_corners(self):
        """Windows 11: smooth rounded corners and a thin border drawn by the system. True if it worked;
        otherwise place_strip clips the window to a rounded shape (Windows 10)."""
        import ctypes
        from ctypes import wintypes as W
        try:
            dwm = ctypes.WinDLL("dwmapi")
            dwm.DwmSetWindowAttribute.argtypes = [W.HWND, W.DWORD, ctypes.c_void_p, W.DWORD]
            DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND, DWMWA_BORDER_COLOR = 33, 2, 34
            pref = ctypes.c_int(DWMWCP_ROUND)
            if dwm.DwmSetWindowAttribute(self.strip_hwnd, DWMWA_WINDOW_CORNER_PREFERENCE,
                                         ctypes.byref(pref), 4) != 0:
                return False
            r, g, b = (int(STRIP_BORDER[i:i + 2], 16) for i in (1, 3, 5))
            color = W.DWORD(r | g << 8 | b << 16)                        # COLORREF is 0x00BBGGRR
            dwm.DwmSetWindowAttribute(self.strip_hwnd, DWMWA_BORDER_COLOR, ctypes.byref(color), 4)
            return True
        except Exception:
            return False

    def clip_round(self, width, height):
        import ctypes
        from ctypes import wintypes as W
        gdi = ctypes.WinDLL("gdi32")
        gdi.CreateRoundRectRgn.argtypes = [ctypes.c_int] * 6
        gdi.CreateRoundRectRgn.restype = W.HANDLE
        self.w.SetWindowRgn.argtypes = [W.HWND, W.HANDLE, W.BOOL]
        d = STRIP_RADIUS * 2
        rgn = gdi.CreateRoundRectRgn(0, 0, width + 1, height + 1, d, d)
        self.w.SetWindowRgn(self.strip_hwnd, rgn, True)              # Windows owns the region now

    def fill(self, items):
        """A vertical list: one row per suggestion (number, word); self.sel is highlighted.
        Mouse: click a row to pick it; press and move anywhere on the list to drag it out of the way."""
        for child in self.strip_box.winfo_children():
            child.destroy()
        self.rows = []
        if any(cmd for _, _, cmd in items):          # top bar: a grip to drag by, and a settings button
            bar = tk.Frame(self.strip_box, bg=STRIP_BG, cursor="fleur")
            bar.pack(side="top", fill="x")
            grip = tk.Label(bar, text="• • •", bg=STRIP_BG, fg=STRIP_NUM, font=(STRIP_FONT[0], 7), cursor="fleur")
            grip.pack(side="left", expand=True, fill="x")
            gear = tk.Label(bar, text="⚙", bg=STRIP_BG, fg=STRIP_NUM, font=(STRIP_FONT[0], 9), cursor="hand2", padx=4)
            gear.pack(side="right")
            for w in (bar, grip):
                self.bind_mouse(w, None)
                w.bind("<Double-Button-1>", self.drag_reset)
            gear.bind("<ButtonRelease-1>", lambda e: (self.hide(), self.app.q.put(self.app.open_settings)))
        for i, (label, hint, cmd) in enumerate(items):
            row = tk.Frame(self.strip_box, bg=STRIP_BG, cursor="hand2" if cmd else "", padx=6, pady=1)
            row.pack(side="top", fill="x")
            parts = []
            if hint:                                   # the key that picks it, before the word
                parts.append(tk.Label(row, text=hint, bg=STRIP_BG, fg=STRIP_NUM, font=STRIP_NUM_FONT,
                                      width=2, anchor="e"))
            parts.append(tk.Label(row, text=label, bg=STRIP_BG, fg=STRIP_FG, font=STRIP_FONT,
                                  padx=6, anchor="w"))
            for w in parts:
                w.pack(side="left")
            if cmd:
                for w in [row] + parts:
                    self.bind_mouse(w, cmd)
            self.rows.append([row] + parts)
        self.paint_selection()

    def bind_mouse(self, widget, pick):
        widget.bind("<ButtonPress-1>", self.drag_start)
        widget.bind("<B1-Motion>", self.drag_move)
        widget.bind("<ButtonRelease-1>", lambda e, p=pick: self.drag_end(e, p))

    # --- dragging the list (the spot is kept, relative to the cursor, while you type in this window)
    def drag_start(self, e):
        self.dragging = [e.x_root, e.y_root, self.strip.winfo_x(), self.strip.winfo_y(), False]

    def drag_move(self, e):
        d = getattr(self, "dragging", None)
        if not d:
            return
        sx, sy, wx, wy, moved = d
        if not moved and abs(e.x_root - sx) + abs(e.y_root - sy) < 5:
            return                                   # a click, not a drag (yet)
        d[4] = True
        x, y = wx + e.x_root - sx, wy + e.y_root - sy
        self.strip.geometry(f"+{x}+{y}")
        w_, h = self.strip.winfo_width(), self.strip.winfo_height()
        self.strip_rect = (x, y, x + w_, y + h)

    def drag_end(self, e, pick=None):
        d = getattr(self, "dragging", None)
        self.dragging = None
        if not d:
            return
        if not d[4]:                                 # released without moving: a click
            if pick:
                pick()
            return
        dx = self.strip.winfo_x() - self.strip_default[0]
        dy = self.strip.winfo_y() - self.strip_default[1]
        self.drag_offset = (self.buf_hwnd, dx, dy)
        if self.ghost_point:                        # the grey text stays at the cursor
            self.place_ghost(self.ghost_point)

    def drag_reset(self, e=None):
        self.drag_offset = None
        if self.visible and getattr(self, "last_point", None):
            self.place_strip(self.last_point)

    def paint_selection(self):
        for i, widgets in enumerate(self.rows):
            bg = STRIP_SEL if (i == self.sel and len(self.rows) > 1) else STRIP_BG
            for w in widgets:
                w.configure(bg=bg)

    def move(self, step):
        """Up/down arrows: move the highlight (wraps around) and update the grey text."""
        if not self.visible or not self.shown:
            return
        self.sel = (self.sel + step) % len(self.shown)
        self.paint_selection()
        if self.ghost_point:
            self.place_ghost(self.ghost_point)

    def place_strip(self, point):
        x, top, bottom = point
        self.strip.update_idletasks()
        w_, h = self.strip.winfo_reqwidth(), self.strip.winfo_reqheight()
        left, mtop, right, mbottom = monitor_bounds(x, top)
        y = bottom + 4 if bottom + 4 + h < mbottom else top - h - 4
        x = max(left, min(x, right - w_))
        self.strip_default, self.last_point = (x, y), point
        off = getattr(self, "drag_offset", None)
        if off and off[0] == self.buf_hwnd:          # you moved it: keep it there, relative to the cursor
            x = max(left, min(x + off[1], right - w_))
            y = max(mtop, min(y + off[2], mbottom - h))
        self.strip_rect = (x, y, x + w_, y + h)
        w = self.w
        before = w.GetForegroundWindow()
        self.strip.geometry(f"+{int(x)}+{int(y)}")
        self.strip.update_idletasks()
        if not self.round_native:
            self.clip_round(w_, h)
        w.ShowWindow(self.strip_hwnd, 4)                             # SW_SHOWNOACTIVATE
        SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x1, 0x2, 0x10
        w.SetWindowPos(self.strip_hwnd, -1, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE)  # topmost
        self.keep_focus(before)
        if not self.strip_checked:                  # once per run: did it really appear?
            self.strip_checked = True
            log(f"predict: strip visible={bool(w.IsWindowVisible(self.strip_hwnd))} at {int(x)},{int(y)} "
                f"size {w_}x{h}, corners={'windows11' if self.round_native else 'clipped'}")

    def keep_focus(self, hwnd):
        """If showing the strip stole focus, hand it straight back so typing keeps working."""
        if hwnd and hwnd != self.strip_hwnd and self.w.GetForegroundWindow() == self.strip_hwnd:
            log("predict: strip took focus, giving it back")
            self.w.SetForegroundWindow(hwnd)

    def show(self, words, point, gen, exact=True):
        if gen != self.gen or not self.strip:
            return
        if self.visible and self.shown_gen == gen and self.sel != 0:
            return                             # you're already choosing: don't reshuffle the list
        try:
            self._show(words, point, gen, exact)
        except Exception:
            log("predict: strip failed:\n" + traceback.format_exc())

    def _show(self, words, point, gen, exact):
        self.sel, self.shown_gen = 0, gen
        self.shown, self.visible = list(words), True
        items = [(w + ("  ✎" if ins.startswith("\b") else ""), str(i + 1),     # ✎ = fixes what you typed
                  lambda i=i: threading.Thread(target=self.accept, args=(self.shown[i],), daemon=True).start())
                 for i, (w, ins) in enumerate(words)]
        self.fill(items)
        self.place_strip(point)
        self.ghost_point = point if (exact and GHOST) else None
        if self.ghost_point:
            self.place_ghost(point)
        else:
            if self.ghost:
                self.w.ShowWindow(self.ghost_hwnd, 0)
            self.ghost_why("off in settings" if not GHOST else
                           "this app doesn't tell Windows where the cursor is (list shown at the mouse)")

    def flash(self, text, point):
        if not self.strip:
            return
        self.sel = 0
        self.fill([(text, "", None)])
        self.shown, self.visible, self.ghost_point = [], True, None
        self.place_strip(point)
        g = self.gen
        self.strip.after(1200, lambda: self.gen == g and self.hide())

    def hide(self):
        was = self.visible
        self.visible, self.shown = False, []
        if was and self.strip:
            def _hide():
                self.w.ShowWindow(self.strip_hwnd, 0)
                if self.ghost:
                    self.w.ShowWindow(self.ghost_hwnd, 0)
            self.app.q.put(_hide)


PRIVACY_NOTE = ("Your data stays on this PC. Suggestions, auto-correct and everything Careter learns run "
                "here and are never sent anywhere (what it learns is encrypted with your Windows login). "
                "Only the Ctrl+Alt+Space assistant goes online: when you use it, the text around your cursor "
                "goes to the AI service you chose in Settings, and nowhere else.")


HELP_HOME = """## Why Careter
• Most apps on a PC have no word suggestions or spelling fixes. Careter adds them everywhere you type, like a phone keyboard does.
• It's free, works offline, learns how you write, and never sends what you type anywhere. What it learns stays encrypted on this PC.
• For anything bigger than a word, press Ctrl+Alt+Space for the AI assistant at your cursor.
"""

HELP_USE = """# Getting started
Careter runs quietly in the background. You don't open it: just type anywhere (Notepad, Word, Gmail, WhatsApp Web, chat apps…) and it helps you type.

## Word suggestions
• Type a few words and pause for a moment. A small dark list appears under the line with up to 5 suggestions.
• The highlighted suggestion also appears as grey text right after your cursor.
• Tab or Enter: insert the grey text. Then it suggests the next word, so you can keep going.
• ↑ / ↓: choose another suggestion (the grey text follows).
• 1 – 5: insert that suggestion directly. You can also click it.
• Esc, or just keep typing: hide the list.
• Mouse: click a suggestion to insert it. In the way? Press and drag the list anywhere (or by its • • • grip). It stays at that spot while you type in that window; double-click the grip to put it back.
• The ⚙ on the list opens these settings (handy if Ctrl+Alt+S is taken on your PC).
• Don't want Enter to pick suggestions? Settings → Typing → switch off "Enter also inserts the suggestion".
• At the start of a sentence, type the first letter and suggestions appear.
• While the list is showing, ↑ ↓, Tab, Enter and 1–5 belong to Careter; the rest of the time they work as usual (press Esc first if you want a new line).
• No grey text? Some apps don't tell Windows where the cursor is. The list still appears (at the mouse), just without grey text.

## Auto-correct
• When you press space after a typo, it's fixed: "teh" → "the", "recieve" → "receive", "definatly" → "definitely".
• Press Backspace right after to put your word back. Careter remembers it and never changes it again.
• Names, CAPITALS, numbers and the first word in a box are never changed.

## Turn it off for a moment
• Double-tap Ctrl (or press Ctrl+Alt+P) to switch suggestions and auto-correct off, for example to type numbers or Manglish. Do it again to switch them back on. A small label shows which.

## The AI assistant
• Press Ctrl+Alt+Space anywhere to open Careter's assistant at your cursor. It reads what's around the cursor, answers, and can take actions after asking you.
• It needs an AI model and its API key: set them in the Settings tab → AI assistant, then click Save & test.
• Only the assistant uses that key; word suggestions run on this PC and never send anything anywhere.

## It learns from you
• Words and phrases you use often move to the top of the list. Personal names are never suggested or learned.
• What it learns stays on this PC, encrypted. Settings tab → Learning: export it or forget everything.

## Keys at a glance
• Tab or Enter: insert the grey text
• ↑ ↓: choose   ·   1–5: pick   ·   Esc: hide the list
• Backspace right after a correction: undo it
• Double-tap Ctrl or Ctrl+Alt+P: suggestions on / off
• Ctrl+Alt+Space: AI assistant
• Ctrl+Alt+S: this window

## Something not working?
• Open the Log tab, try again, click Copy and send the log to whoever helps you. It never contains what you typed.
"""

HELP_INSTALL = r"""# Install on Windows
1. Open PowerShell: press the Windows key, type PowerShell, press Enter.
2. Paste this line and press Enter:
$ irm https://raw.githubusercontent.com/pranavmtn/careter/main/install.ps1 | iex
3. That's all: no API key needed. Word suggestions, grey text and auto-correct work without one; only the Ctrl+Alt+Space assistant needs a key, which you add any time in Settings → AI assistant (OpenAI, Claude and others work).
That's it: Careter starts now and every time you sign in. Press Ctrl+Alt+S to see this window.

## Update to the newest version
Always stop Careter first, then update, then start it again. Run these one by one in PowerShell:
$ cd "{dir}"
$ .\.venv\Scripts\python caret_ai.py --stop
$ git pull
$ .\.venv\Scripts\python caret_ai.py --autostart on

## Browser helper (Chrome, Edge, Brave)
Puts the suggestions exactly under your text on websites.
1. In Chrome open chrome://extensions (in Edge: edge://extensions).
2. Switch on Developer mode (top right).
3. Click Load unpacked and choose this folder:
$ {dir}\chrome-extension
4. Reload the tabs you had open.

## Careter for Chrome (on computers without Careter)
Suggestions, auto-correct and learning inside the browser only, no install needed. Same steps as above, but choose this folder instead:
$ {dir}\careter-chrome
On a PC where Careter is running, this extension stays quiet and Careter does the work.

## Handy commands
Run them in PowerShell after cd "{dir}":
$ .\.venv\Scripts\python caret_ai.py --stop
• closes Careter (it comes back within 5 minutes if start-with-Windows is on)
$ .\.venv\Scripts\python caret_ai.py --autostart on
• start with Windows and keep running (also starts it now); use off to stop that
$ .\.venv\Scripts\python caret_ai.py
• run it in the PowerShell window to watch its messages (stop it first)
$ .\.venv\Scripts\python caret_ai.py --settings
• open this window

## Uninstall
$ .\.venv\Scripts\python caret_ai.py --autostart off
$ .\.venv\Scripts\python caret_ai.py --stop
Then delete the folder {dir} and remove the Careter extensions in chrome://extensions.
"""


def set_env_values(values):
    """Write KEY=VALUE pairs into the .env next to this script (a None value removes the key)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    done = set()
    out = []
    for line in lines:
        k = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if k in values:
            done.add(k)
            if values[k] is not None:
                out.append(f"{k}={values[k]}")
            continue
        out.append(line)
    out += [f"{k}={v}" for k, v in values.items() if k not in done and v is not None]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    for k, v in values.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


TAGLINE = "Makes you a faster typer."
SHARE_INSTALL = "irm https://raw.githubusercontent.com/pranavmtn/careter/main/install.ps1 | iex"
SHARE_URL = "https://github.com/pranavmtn/careter"
SHARE_POSTS = {
    "LinkedIn": f"""Hey 👋 Found a little tool that has quietly cut down my keystrokes, and the time I spend typing, every single day. It's called Careter, and it makes you a faster typer.

Phones finished our words years ago. On a PC, most apps still make you type every letter. Careter fixes that in every Windows app: Notepad, Word, Gmail, WhatsApp Web, chat tools, anywhere.

✨ What I love about it
• The next word shows up as grey text right after the cursor. Tab, and it's typed
• A small list of 5 options under the line: ↑ ↓ or 1–5 to pick
• Typos fixed as I hit space ("teh" → "the"), one Backspace to undo
• It learns the words and phrases I use most
• Ctrl+Alt+Space opens an AI assistant right at my cursor

🔒 Private by design: suggestions, auto-correct and what it learns stay on my PC (encrypted). Only the assistant goes online, and only to the AI service I pick.

⚡ Install on Windows (one line in PowerShell):
{SHARE_INSTALL}

Free, code on GitHub: {SHARE_URL}

Give it a day and see how many keystrokes it saves you 🙌

#productivity #AI #Windows #techtools""",
    "X": f"""Hey 👋 Found Careter. It makes you a faster typer, in every Windows app.

✨ Grey-text autocomplete (Tab)
✅ Auto-correct on space
🧠 Learns your words
🔒 Typing stays on your PC

Install (PowerShell):
{SHARE_INSTALL}

github.com/pranavmtn/careter""",
    "Instagram": """Hey 👋 Found a little tool that has quietly cut my keystrokes, and my typing time, every single day ⌨️⚡

It's called Careter, and it makes you a faster typer: phone-style word suggestions for every app on a Windows PC.

✨ Next word appears in grey, press Tab and it's typed
✅ Typos fixed as you hit space
🧠 Learns the words YOU use
🤖 Ctrl+Alt+Space = AI assistant at your cursor
🔒 Your typing never leaves your PC

Free to install: one line in PowerShell. Steps at github.com/pranavmtn/careter 🔗

#productivity #techtools #windows #ai #typing""",
}
# where "Copy & open" goes; X can take the text in the link, the others get it from the clipboard
SHARE_OPEN = {
    "LinkedIn": "https://www.linkedin.com/feed/",
    "X": "https://x.com/intent/tweet?text={text}",
    "Instagram": "https://www.instagram.com/",
}


SHORTCUTS = [
    ("Tab  or  Enter", "insert the suggestion (the grey text)"),
    ("↑  ↓", "choose another suggestion"),
    ("1 – 5", "pick a suggestion directly"),
    ("Esc", "hide the suggestions"),
    ("Backspace", "right after an auto-correct: undo it"),
    ("Ctrl  Ctrl", "double-tap: suggestions on / off"),
    ("Ctrl+Alt+Space", "AI assistant at your cursor"),
    ("Ctrl+Alt+S  or  ⚙", "settings, help and your AI key (⚙ is on the list)"),
]


class ShortcutsCard:
    """Shown once on first start (and from Settings → Home): Careter is running, and these are the keys."""

    def __init__(self, app, first=False):
        global WELCOMED
        self.app = app
        win = self.win = tk.Toplevel(app.root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.configure(bg=S_LINE)
        box = tk.Frame(win, bg=S_BG, padx=20, pady=16)
        box.pack(padx=1, pady=1)
        tk.Label(box, text="Careter is running ✓" if first else "Careter shortcuts", bg=S_BG, fg=S_FG,
                 font=(S_FONT[0], S_FONT[1] + 4, "bold"), anchor="w").pack(fill="x")
        tk.Label(box, text=TAGLINE + " Type anywhere and pause: suggestions appear under the line.",
                 bg=S_BG, fg=S_MUTED, font=S_SMALL, anchor="w", justify="left", wraplength=380).pack(fill="x", pady=(2, 10))
        grid = tk.Frame(box, bg=S_BG)
        grid.pack(fill="x")
        for i, (keys, what) in enumerate(SHORTCUTS):
            tk.Label(grid, text=keys, bg=S_FIELD, fg=S_FG, font=(S_FONT[0], S_FONT[1] - 1, "bold"),
                     padx=8, pady=2).grid(row=i, column=0, sticky="w", pady=2)
            tk.Label(grid, text=what, bg=S_BG, fg=S_FG, font=S_SMALL, anchor="w").grid(row=i, column=1, sticky="w",
                                                                                     padx=(10, 0))
        if not has_ai_key():
            tk.Label(box, text="The AI assistant needs a key: add it any time in Ctrl+Alt+S → Settings → AI assistant.",
                     bg=S_BG, fg=S_MUTED, font=S_SMALL, anchor="w", justify="left", wraplength=380).pack(fill="x", pady=(10, 0))
        row = tk.Frame(box, bg=S_BG)
        row.pack(fill="x", pady=(14, 0))
        tk.Button(row, text="Got it", command=self.close, bg=S_DARK, fg="#ffffff", activebackground="#374151",
                  activeforeground="#ffffff", relief="flat", bd=0, font=S_BOLD, padx=16, pady=5,
                  cursor="hand2").pack(side="right")
        tk.Button(row, text="Open settings", command=lambda: (self.close(), app.open_settings()), bg=S_FIELD,
                  fg=S_FG, activebackground=S_LINE, relief="flat", font=S_FONT, padx=12, pady=4,
                  cursor="hand2").pack(side="right", padx=8)
        win.bind("<Escape>", lambda e: self.close())
        win.update_idletasks()
        w, h = win.winfo_reqwidth(), win.winfo_reqheight()
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        win.geometry(f"+{sw - w - 24}+{sh - h - 72}")        # bottom-right, above the taskbar
        if IS_WIN:
            win.after(10, lambda: round_window(app_hwnd(win), S_LINE))
        win.focus_force()
        if first:
            WELCOMED = True
            save_settings()

    def close(self):
        try:
            self.win.destroy()
        except tk.TclError:
            pass


def app_hwnd(win):
    import ctypes
    return ctypes.windll.user32.GetAncestor(win.winfo_id(), 2)


class SettingsWindow:
    """Careter Settings (Ctrl+Alt+S): tabs for the switches, a how-to for new users, install and update
    steps, and the live log."""

    TABS = ("Home", "Settings", "How to use", "Install & update", "Share", "Log")

    def __init__(self, app):
        self.app = app
        win = self.win = tk.Toplevel(app.root)
        win.title("Careter Settings")
        win.configure(bg=S_BG, padx=24, pady=18)
        win.attributes("-topmost", True)
        win.minsize(720, 640)
        self.vars = {}
        here = os.path.dirname(os.path.abspath(__file__))

        top = tk.Frame(win, bg=S_BG)
        top.pack(fill="x")
        tk.Label(top, text="Careter", bg=S_BG, fg=S_FG, font=(S_FONT[0], S_FONT[1] + 9, "bold"),
                 anchor="w").pack(side="left")
        tk.Label(top, text="  " + TAGLINE, bg=S_BG, fg=S_MUTED, font=S_FONT,
                 anchor="sw").pack(side="left", pady=(10, 0))
        bar = tk.Frame(win, bg=S_BG)
        bar.pack(fill="x", pady=(14, 0))
        self.tab_buttons, self.pages = {}, {}
        for name in self.TABS:                      # underline tabs: the chosen one is black + underlined
            cell = tk.Frame(bar, bg=S_BG, cursor="hand2")
            cell.pack(side="left", padx=(0, 22))
            lbl = tk.Label(cell, text=name, bg=S_BG, fg=S_MUTED, font=S_BOLD, pady=4, cursor="hand2")
            lbl.pack()
            line = tk.Frame(cell, bg=S_BG, height=2)
            line.pack(fill="x")
            for w in (cell, lbl):
                w.bind("<Button-1>", lambda e, n=name: self.show_tab(n))
            self.tab_buttons[name] = (lbl, line)
        tk.Frame(win, bg=S_LINE, height=1).pack(fill="x")
        body = tk.Frame(win, bg=S_BG)
        body.pack(fill="both", expand=True, pady=(10, 0))
        for name in self.TABS:
            self.pages[name] = tk.Frame(body, bg=S_BG)

        # the status line and buttons are created first so every tab can use them
        self.bottom = tk.Frame(win, bg=S_BG)
        self.status = tk.Label(self.bottom, text="", bg=S_BG, fg=S_MUTED, font=S_SMALL, anchor="w")
        self.status.pack(fill="x", pady=(8, 0))
        row = tk.Frame(self.bottom, bg=S_BG)
        row.pack(fill="x", pady=(6, 0))
        self.button(row, "Close", self.close).pack(side="right")
        self.restart_btn = self.button(row, "Restart Careter", self.restart)
        self.bottom.pack(fill="x", side="bottom")

        self.build_home(self.pages["Home"])
        self.build_settings(self.pages["Settings"])
        self.build_guide(self.pages["How to use"], HELP_USE)
        self.build_guide(self.pages["Install & update"], HELP_INSTALL.replace("{dir}", here))
        self.build_share(self.pages["Share"])
        self.build_log(self.pages["Log"])
        win.protocol("WM_DELETE_WINDOW", self.close)
        self.show_tab("Home")
        self.focus()

    def show_tab(self, name):
        for n, page in self.pages.items():
            page.pack_forget()
            lbl, line = self.tab_buttons[n]
            lbl.configure(fg=S_FG if n == name else S_MUTED)
            line.configure(bg=S_DARK if n == name else S_BG)
        self.pages[name].pack(fill="both", expand=True)
        if name == "Log":
            self.reload_log()
        if name == "Home":
            self.refresh_home()

    # --- tab: home
    def build_home(self, page):
        self.welcome = tk.Label(page, text="", bg=S_BG, fg=S_FG, font=(S_FONT[0], S_FONT[1] + 5, "bold"),
                                anchor="w")
        self.welcome.pack(fill="x")
        tk.Label(page, text=TAGLINE + " Word suggestions, grey-text autocomplete and auto-correct in every app, plus an AI "
                 "assistant at your cursor. Everything for typing runs on this PC.", bg=S_BG, fg=S_MUTED,
                 font=S_FONT, anchor="w", justify="left", wraplength=660).pack(fill="x", pady=(2, 12))
        cards = tk.Frame(page, bg=S_BG)
        cards.pack(fill="x")
        self.cards = {}
        for i, (key, title) in enumerate([("predict", "Suggestions"), ("autocorrect", "Auto-correct"),
                                          ("learned", "Words learned"), ("model", "Assistant model")]):
            card = tk.Frame(cards, bg=S_SOFT, highlightthickness=1, highlightbackground=S_LINE, padx=14, pady=10)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 10, 0))
            cards.grid_columnconfigure(i, weight=1, uniform="cards")
            tk.Label(card, text=title.upper(), bg=S_SOFT, fg=S_MUTED, font=(S_FONT[0], S_FONT[1] - 2, "bold"),
                     anchor="w").pack(fill="x")
            value = tk.Label(card, text="…", bg=S_SOFT, fg=S_FG, font=(S_FONT[0], S_FONT[1] + 4, "bold"), anchor="w")
            value.pack(fill="x", pady=(4, 0))
            self.cards[key] = value
        row = tk.Frame(page, bg=S_BG)
        row.pack(fill="x", pady=(14, 4))
        self.button(row, "How to use", lambda: self.show_tab("How to use"), primary=True).pack(side="left")
        self.button(row, "Settings", lambda: self.show_tab("Settings")).pack(side="left", padx=8)
        self.button(row, "Install & update", lambda: self.show_tab("Install & update")).pack(side="left")
        self.button(row, "Share Careter", lambda: self.show_tab("Share")).pack(side="left", padx=8)
        self.button(row, "Shortcuts", lambda: ShortcutsCard(self.app)).pack(side="left")
        tk.Label(row, text="Ctrl+Alt+S opens this window", bg=S_BG, fg=S_MUTED, font=S_SMALL).pack(side="right")
        privacy = tk.Frame(page, bg=S_SOFT, highlightthickness=1, highlightbackground=S_LINE, padx=14, pady=10)
        privacy.pack(fill="x", pady=(10, 0))
        tk.Label(privacy, text="🔒  PRIVATE BY DESIGN", bg=S_SOFT, fg=S_MUTED,
                 font=(S_FONT[0], S_FONT[1] - 2, "bold"), anchor="w").pack(fill="x")
        tk.Label(privacy, text=PRIVACY_NOTE, bg=S_SOFT, fg=S_FG, font=S_FONT, anchor="w", justify="left",
                 wraplength=640).pack(fill="x", pady=(4, 0))
        tk.Frame(page, bg=S_LINE, height=1).pack(fill="x", pady=(10, 0))
        story = tk.Frame(page, bg=S_BG)
        story.pack(fill="both", expand=True)
        self.build_guide(story, HELP_HOME)

    def refresh_home(self):
        self.welcome.configure(text=f"Welcome, {USER_NAME}" if USER_NAME else "Welcome to Careter")
        pred = self.app.predictions
        on = lambda b: "On" if b else "Off"
        self.cards["predict"].configure(text=on(pred.enabled if pred else PREDICT))
        self.cards["autocorrect"].configure(text=on(AUTOCORRECT and bool(pred and pred.speller)))
        n = len(pred.user_model.uni) if (pred and pred.user_model) else 0
        self.cards["learned"].configure(text=f"{n:,}" if LEARN else "Off")
        self.cards["model"].configure(text=(MODEL if len(MODEL) <= 16 else MODEL[:15] + "…") if has_ai_key()
                                      else "No key yet")

    # --- tab: settings (scrollable: it's a long list)
    def build_settings(self, outer):
        canvas = tk.Canvas(outer, bg=S_BG, highlightthickness=0)
        scroll = tk.Scrollbar(outer, command=canvas.yview)
        page = tk.Frame(canvas, bg=S_BG)
        page.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=page, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        wheel = lambda e: canvas.yview_scroll(int(-e.delta / 120), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", wheel))   # only while the mouse
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))       # is over this list
        pred = self.app.predictions

        def header(text):
            tk.Frame(page, bg=S_LINE, height=1).pack(fill="x", pady=(14, 0))
            tk.Label(page, text=text.upper(), bg=S_BG, fg=S_MUTED, font=(S_FONT[0], S_FONT[1] - 2, "bold"),
                     anchor="w").pack(fill="x", pady=(8, 4))

        def switch(key, text, value, note=""):
            v = tk.BooleanVar(value=value)
            self.vars[key] = v
            row = tk.Frame(page, bg=S_BG)
            row.pack(fill="x")
            tk.Checkbutton(row, text=text, variable=v, command=lambda: self.changed(key), bg=S_BG, fg=S_FG,
                           selectcolor=S_BG, activebackground=S_BG, activeforeground=S_FG, font=S_FONT,
                           anchor="w").pack(side="left")
            if note:
                tk.Label(row, text=note, bg=S_BG, fg=S_MUTED, font=S_SMALL).pack(side="left", padx=6)

        header("You")
        row = tk.Frame(page, bg=S_BG)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="Your name", bg=S_BG, fg=S_FG, font=S_FONT, width=14, anchor="w").pack(side="left")
        self.name_entry = tk.Entry(row, bg=S_SOFT, fg=S_FG, insertbackground=S_FG, relief="flat", font=S_FONT,
                                   width=30, highlightthickness=1, highlightbackground=S_LINE, highlightcolor=S_DARK)
        self.name_entry.insert(0, USER_NAME)
        self.name_entry.pack(side="left", ipady=3)
        self.name_entry.bind("<Return>", lambda e: self.save_name())
        self.button(row, "Save", self.save_name, primary=True).pack(side="left", padx=8)
        tk.Label(page, text="Used to greet you, and the assistant signs drafts with it. Stays on this PC.",
                 bg=S_BG, fg=S_MUTED, font=S_SMALL, anchor="w").pack(fill="x")
        tk.Label(page, text="🔒  " + PRIVACY_NOTE, bg=S_BG, fg=S_MUTED, font=S_SMALL, anchor="w",
                 justify="left", wraplength=640).pack(fill="x", pady=(6, 0))

        header("Typing")
        switch("predict", "Word suggestions while I type", pred.enabled if pred else PREDICT,
               "(also: double-tap Ctrl)")
        switch("ghost", "Grey text after the cursor (Tab inserts it)", GHOST)
        switch("enter", "Enter also inserts the suggestion", ENTER_PICKS, "(off: Enter always makes a new line)")
        row = tk.Frame(page, bg=S_BG)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="      Suggestions in the list:", bg=S_BG, fg=S_FG, font=S_FONT).pack(side="left")
        self.count = tk.IntVar(value=PREDICT_COUNT)
        for n in (3, 4, 5):
            tk.Radiobutton(row, text=str(n), value=n, variable=self.count, command=lambda: self.changed("count"),
                           bg=S_BG, fg=S_FG, selectcolor=S_BG, activebackground=S_BG, font=S_FONT).pack(side="left")
        switch("autocorrect", "Auto-correct typos when I press space", AUTOCORRECT, "(Backspace right after undoes)")
        switch("terminals", "Also suggest in terminals (PowerShell, cmd)", PREDICT_IN_TERMINALS)
        header("Smarter suggestions")
        switch("engine", "Use the local AI model (SmolLM2)", PREDICT_ENGINE == "llm", "(applies after a restart)")

        header("AI assistant  (Ctrl+Alt+Space)")
        self.ai = {}

        def field(label, key, value, secret=False, hint=""):
            row = tk.Frame(page, bg=S_BG)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, bg=S_BG, fg=S_FG, font=S_FONT, width=14, anchor="w").pack(side="left")
            e = tk.Entry(row, bg=S_SOFT, fg=S_FG, insertbackground=S_FG, relief="flat", font=S_FONT, width=44,
                         show="•" if secret else "", highlightthickness=1, highlightbackground=S_LINE,
                         highlightcolor=S_DARK)
            e.insert(0, value)
            e.pack(side="left", ipady=3)
            self.ai[key] = e
            if hint:
                tk.Label(row, text=hint, bg=S_BG, fg=S_MUTED, font=S_SMALL).pack(side="left", padx=6)
            return e

        field("Model name", "model", MODEL, hint="e.g. gpt-6-luna")
        key_entry = field("API key", "key", os.environ.get("OPENAI_API_KEY", ""), secret=True)
        self.show_key = tk.BooleanVar(value=False)
        tk.Checkbutton(key_entry.master, text="Show", variable=self.show_key, bg=S_BG, fg=S_FG, selectcolor=S_BG,
                       activebackground=S_BG, activeforeground=S_FG, font=S_FONT,
                       command=lambda: key_entry.configure(show="" if self.show_key.get() else "•")).pack(side="left", padx=6)
        field("API address", "base", os.environ.get("OPENAI_BASE_URL", ""),
              hint="empty = OpenAI")
        tk.Label(page, text="      Any OpenAI-compatible service works: e.g. OpenRouter https://openrouter.ai/api/v1, "
                 "Groq https://api.groq.com/openai/v1, Ollama http://localhost:11434/v1",
                 bg=S_BG, fg=S_MUTED, font=S_SMALL, anchor="w", justify="left",
                 wraplength=620).pack(fill="x")
        row = tk.Frame(page, bg=S_BG)
        row.pack(fill="x", pady=(4, 0))
        self.button(row, "Save & test", self.save_ai, primary=True).pack(side="left")
        tk.Label(row, text="saved in the .env file in the Careter folder", bg=S_BG, fg=S_MUTED,
                 font=S_SMALL).pack(side="left", padx=8)

        header("My words")
        tk.Label(page, text="Words you use that it should always offer and never auto-correct "
                 "(product names, tech terms, Manglish…). Separate with spaces or commas.",
                 bg=S_BG, fg=S_MUTED, font=S_SMALL, anchor="w", justify="left", wraplength=640).pack(fill="x")
        self.mine_text = tk.Text(page, height=3, width=70, bg=S_SOFT, fg=S_FG, insertbackground=S_FG,
                                 relief="flat", font=S_FONT, wrap="word", highlightthickness=1,
                                 highlightbackground=S_LINE, highlightcolor=S_DARK)
        um = pred.user_model if pred else None
        self.mine_text.insert("1.0", " ".join(sorted(um.mine)) if um else "")
        self.mine_text.pack(fill="x", pady=(4, 4))
        row = tk.Frame(page, bg=S_BG)
        row.pack(fill="x")
        self.button(row, "Save my words", self.save_mine, primary=True).pack(side="left")

        header("Learning  (stays on this PC, encrypted)")
        switch("learn", "Learn my words and phrases", LEARN)
        row = tk.Frame(page, bg=S_BG)
        row.pack(fill="x", pady=(4, 0))
        self.button(row, "Export what it learned…", self.export).pack(side="left")
        self.button(row, "Forget everything", self.forget).pack(side="left", padx=8)
        header("Startup")
        switch("autostart", "Start with Windows and keep running", self.autostart_on() if IS_WIN else False)
        tk.Label(page, text="New here? See the How to use tab.", bg=S_BG, fg=S_MUTED,
                 font=S_SMALL, anchor="w").pack(fill="x", pady=(14, 4))

    def save_name(self):
        global USER_NAME
        USER_NAME = " ".join(self.name_entry.get().split())[:40]
        save_settings()
        self.say(f"Saved. Hi {USER_NAME}!" if USER_NAME else "Name cleared.")

    def save_mine(self):
        pred = self.app.predictions
        if not pred or not pred.user_model:
            self.say("Turn on learning (and restart) to keep your own words.")
            return
        words = [w for w in re.split(r"[\s,;]+", self.mine_text.get("1.0", "end")) if w]
        pred.user_model.set_mine(words)
        pred.save_learned()
        self.say(f"Saved {len(pred.user_model.mine)} words.")
        log(f"settings: my words saved ({len(pred.user_model.mine)})")

    def save_ai(self):
        """Save the assistant's model, key and address, switch to them now, and test them."""
        global MODEL, client
        model = self.ai["model"].get().strip()
        key = self.ai["key"].get().strip()
        base = self.ai["base"].get().strip().rstrip("/")
        if not model or not key:
            self.say("Enter a model name and an API key.")
            return
        set_env_values({"CARET_AI_MODEL": model, "OPENAI_API_KEY": key, "OPENAI_BASE_URL": base or None})
        MODEL = model
        client = OpenAI(api_key=key, base_url=base or None)
        self.say(f"Saved. Testing {model}…")

        def test():
            try:
                client.chat.completions.create(model=model, max_tokens=5,
                                               messages=[{"role": "user", "content": "Say OK"}])
                msg = f"Works: the assistant now uses {model}."
                log(f"settings: assistant model set to {model}, test OK")
            except Exception as e:
                msg = f"Saved, but the test failed: {str(e)[:160]}"
                log(f"settings: assistant test failed for {model}: {str(e)[:200]}")
            self.app.q.put(lambda: self.alive() and self.say(msg))
        threading.Thread(target=test, daemon=True).start()

    # --- tabs: how to use, install & update
    def build_guide(self, page, content):
        """Show a guide: '# ' headings, '## ' subheadings, '$ ' commands (click one to copy it)."""
        t = tk.Text(page, bg=S_BG, fg=S_FG, relief="flat", wrap="word", font=S_FONT, padx=4, pady=2,
                    cursor="arrow", highlightthickness=0, spacing1=2, spacing3=2)
        scroll = tk.Scrollbar(page, command=t.yview)
        t.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        t.pack(side="left", fill="both", expand=True)
        mono = ("Consolas", S_FONT[1]) if IS_WIN else ("Menlo", S_FONT[1] + 1)
        t.tag_configure("h1", font=(S_FONT[0], S_FONT[1] + 4, "bold"), foreground=S_FG, spacing1=4, spacing3=6)
        t.tag_configure("h2", font=(S_FONT[0], S_FONT[1] + 1, "bold"), foreground=S_FG, spacing1=14, spacing3=4)
        t.tag_configure("code", font=mono, background=S_FIELD, foreground=S_FG, lmargin1=12, lmargin2=12,
                        rmargin=12, spacing1=3, spacing3=3)
        t.tag_configure("bullet", lmargin1=8, lmargin2=22)
        t.tag_configure("hint", foreground=S_MUTED, font=S_SMALL)
        t.tag_bind("code", "<Enter>", lambda e: t.configure(cursor="hand2"))
        t.tag_bind("code", "<Leave>", lambda e: t.configure(cursor="arrow"))
        t.tag_bind("code", "<Button-1>", lambda e: self.copy_code(t, e))
        for line in content.strip("\n").split("\n"):
            if line.startswith("# "):
                t.insert("end", line[2:] + "\n", "h1")
            elif line.startswith("## "):
                t.insert("end", line[3:] + "\n", "h2")
            elif line.startswith("$ "):
                t.insert("end", line[2:] + "\n", "code")
            elif line.startswith("• "):
                t.insert("end", line + "\n", "bullet")
            else:
                t.insert("end", line + "\n")
        if "\n$ " in content:
            t.insert("end", "\nTip: click a command line to copy it.\n", "hint")
        t.configure(state="disabled")

    def copy_code(self, t, event):
        index = t.index(f"@{event.x},{event.y}")
        line = t.get(f"{index} linestart", f"{index} lineend").strip()
        self.win.clipboard_clear()
        self.win.clipboard_append(line)
        self.say(f"Copied: {line}")

    # --- tab: share
    def build_share(self, outer):
        canvas = tk.Canvas(outer, bg=S_BG, highlightthickness=0)
        scroll = tk.Scrollbar(outer, command=canvas.yview)
        page = tk.Frame(canvas, bg=S_BG)
        page.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=page, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        wheel = lambda e: canvas.yview_scroll(int(-e.delta / 120), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        tk.Label(page, text="Share Careter", bg=S_BG, fg=S_FG, font=(S_FONT[0], S_FONT[1] + 5, "bold"),
                 anchor="w").pack(fill="x")
        tk.Label(page, text="Like it? Tell your friends. Pick a platform: the post is copied for you, so just "
                 "paste it (or make it your own) and share.", bg=S_BG, fg=S_MUTED, font=S_FONT, anchor="w",
                 justify="left", wraplength=640).pack(fill="x", pady=(2, 8))
        for name, text in SHARE_POSTS.items():
            card = tk.Frame(page, bg=S_SOFT, highlightthickness=1, highlightbackground=S_LINE, padx=14, pady=10)
            card.pack(fill="x", pady=(8, 0), padx=(0, 6))
            head = tk.Frame(card, bg=S_SOFT)
            head.pack(fill="x")
            tk.Label(head, text=name, bg=S_SOFT, fg=S_FG, font=(S_FONT[0], S_FONT[1] + 1, "bold")).pack(side="left")
            if name == "X":
                tk.Label(head, text=f"  {x_length(text)}/280", bg=S_SOFT, fg=S_MUTED, font=S_SMALL).pack(side="left")
            self.button(head, f"Copy & open {name}", lambda n=name: self.share(n, True), primary=True).pack(side="right")
            self.button(head, "Copy", lambda n=name: self.share(n, False)).pack(side="right", padx=6)
            lines = min(12, text.count("\n") + 2)
            t = tk.Text(card, height=lines, width=78, bg=S_BG, fg=S_FG, relief="flat", wrap="word", font=S_SMALL,
                        highlightthickness=1, highlightbackground=S_LINE, padx=8, pady=6)
            t.insert("1.0", text)
            t.configure(state="disabled")
            t.pack(fill="x", pady=(8, 0))

    def share(self, name, open_site):
        import urllib.parse
        import webbrowser
        text = SHARE_POSTS[name]
        self.win.clipboard_clear()
        self.win.clipboard_append(text)
        if not open_site:
            self.say(f"{name} post copied. Paste it wherever you like.")
            return
        webbrowser.open(SHARE_OPEN[name].replace("{text}", urllib.parse.quote(text)))
        self.say(f"Opening {name}: the post is filled in, check it and post." if name == "X" else
                 f"Opening {name}: the post is copied, paste it with Ctrl+V and share.")
        log(f"settings: share via {name}")

    # --- tab: log
    def build_log(self, page):
        """What Careter is doing, live (never what you typed)."""
        bar = tk.Frame(page, bg=S_BG)
        bar.pack(fill="x")
        self.only_predict = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="Only word prediction", variable=self.only_predict, command=self.reload_log,
                       bg=S_BG, fg=S_FG, selectcolor=S_BG, activebackground=S_BG, activeforeground=S_FG,
                       font=S_FONT).pack(side="left")
        self.button(bar, "Open log file", self.open_log).pack(side="right")
        self.button(bar, "Copy", self.copy_log).pack(side="right", padx=6)
        box = tk.Frame(page, bg=S_BG)
        box.pack(fill="both", expand=True, pady=(6, 0))
        self.log_text = tk.Text(box, height=20, width=88, bg=S_SOFT, fg=S_FG, insertbackground=S_FG, relief="flat",
                                highlightthickness=1, highlightbackground=S_LINE,
                                font=("Consolas", 9) if IS_WIN else ("Menlo", 10), wrap="none")
        scroll = tk.Scrollbar(box, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)
        self.reload_log()
        self.on_log = lambda line: self.app.q.put(lambda: self.add_log(line))   # log() runs on any thread
        LOG_LISTENERS.append(self.on_log)

    def wanted(self, line):
        return not self.only_predict.get() or "predict" in line or "settings:" in line

    def reload_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        for line in LOG_LINES:
            if self.wanted(line):
                self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def add_log(self, line):
        if not self.alive() or not self.wanted(line):
            return
        at_bottom = self.log_text.yview()[1] > 0.98
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        if int(self.log_text.index("end-1c").split(".")[0]) > 500:
            self.log_text.delete("1.0", "2.0")
        if at_bottom:
            self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def copy_log(self):
        self.win.clipboard_clear()
        self.win.clipboard_append(self.log_text.get("1.0", "end-1c"))
        self.say("Log copied: paste it to Claude to diagnose anything.")

    def open_log(self):
        try:
            os.startfile(LOG_PATH) if IS_WIN else None
        except OSError:
            self.say(f"Log file: {LOG_PATH}")

    def close(self):
        if self.on_log in LOG_LISTENERS:
            LOG_LISTENERS.remove(self.on_log)
        self.win.destroy()

    def button(self, parent, text, cmd, primary=False):
        if primary:
            return tk.Button(parent, text=text, command=cmd, bg=S_DARK, fg="#ffffff", activebackground="#374151",
                             activeforeground="#ffffff", relief="flat", bd=0, font=S_BOLD, padx=14, pady=5,
                             cursor="hand2")
        return tk.Button(parent, text=text, command=cmd, bg=S_FIELD, fg=S_FG, activebackground=S_LINE,
                         activeforeground=S_FG, relief="flat", font=S_FONT, padx=10, pady=3, cursor="hand2")

    def alive(self):
        try:
            return bool(self.win.winfo_exists())
        except tk.TclError:
            return False

    def focus(self):
        self.win.deiconify()
        self.win.lift()
        self.win.focus_force()

    def say(self, text):
        self.status.configure(text=text)

    @staticmethod
    def autostart_on():
        import subprocess
        try:
            return subprocess.run(["schtasks", "/Query", "/TN", TASK_NAME], capture_output=True).returncode == 0
        except OSError:
            return False

    def changed(self, key):
        global PREDICT, GHOST, PREDICT_COUNT, AUTOCORRECT, LEARN, PREDICT_ENGINE, PREDICT_IN_TERMINALS, ENTER_PICKS
        pred = self.app.predictions
        if key == "count":
            PREDICT_COUNT = self.count.get()
        elif key == "autostart":
            on = self.vars[key].get()
            self.say("Setting up auto-start…")
            threading.Thread(target=lambda: (autostart_command("on" if on else "off"),
                                             self.app.q.put(lambda: self.say(
                                                 "Auto-start is " + ("on." if on else "off.")))),
                             daemon=True).start()
            return
        else:
            v = self.vars[key].get()
            if key == "predict":
                PREDICT = v
                if pred and pred.enabled != v:
                    pred.toggle()
            elif key == "ghost":
                GHOST = v
            elif key == "enter":
                ENTER_PICKS = v
            elif key == "autocorrect":
                AUTOCORRECT = v
            elif key == "learn":
                LEARN = v
                if v and pred and not pred.user_model:
                    self.show_restart("Learning starts after a restart.")
            elif key == "terminals":
                PREDICT_IN_TERMINALS = v
            elif key == "engine":
                PREDICT_ENGINE = "llm" if v else "ngram"
                self.show_restart("The AI model setting applies after a restart.")
        save_settings()
        if key not in ("engine",) and not self.status.cget("text").startswith("The AI"):
            self.say("Saved.")

    def show_restart(self, text):
        self.say(text)
        self.restart_btn.pack(side="right", padx=8)

    def export(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(parent=self.win, defaultextension=".json",
                                            initialfile="careter_learned.json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        pred = self.app.predictions
        if pred and pred.user_model:
            pred.save_learned()
        learned_data_command(["--export-learned", path])
        self.say(f"Exported to {path}. It's plain text: delete it when you're done.")

    def forget(self):
        from tkinter import messagebox
        if not messagebox.askyesno("Forget everything?", "Delete all the words and phrases Careter learned "
                                   "from your typing? This can't be undone.", parent=self.win):
            return
        pred = self.app.predictions
        if pred and pred.user_model:
            with pred.user_model.lock:
                pred.user_model.uni.clear()
                pred.user_model.bi.clear()
                pred.user_model.dirty = False
        if pred and pred.speller:
            pred.speller.user.clear()
        learned_data_command(["--forget-learned"])
        self.say("Forgot everything it learned.")

    def restart(self):
        """Start a fresh Careter in 2 seconds (after this one has let go of its lock), then quit."""
        import subprocess
        pyw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
        pyw = pyw if os.path.exists(pyw) else sys.executable
        script = os.path.abspath(__file__)
        code = (f"import time,subprocess;time.sleep(2);"
                f"subprocess.Popen([{pyw!r},{script!r}],cwd={os.path.dirname(script)!r})")
        flags = 0x00000008 | 0x00000200 if IS_WIN else 0          # DETACHED_PROCESS | NEW_PROCESS_GROUP
        subprocess.Popen([pyw, "-c", code], creationflags=flags, close_fds=True)
        pred = self.app.predictions
        if pred:
            pred.save_learned()
        os._exit(0)


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.q = queue.Queue()
        self.session = None
        self.triggering = False
        self.last_trigger = 0.0
        self.log_start = 0
        self.root.after(15, self.pump)
        self.predictions = None
        if IS_WIN:
            try:
                self.predictions = Predictions(self)
            except Exception:
                log("predict: could not start:\n" + traceback.format_exc())
            # OS-level hotkeys: exact combos only, never fire from normal typing or clicks
            hotkeys = {1: (0x2 | 0x1, 0x20, self.on_hotkey)}                  # Ctrl+Alt+Space
            if self.predictions:
                hotkeys[2] = (0x2 | 0x1, ord("P"), self.predictions.toggle)    # Ctrl+Alt+P
                self.double_ctrl = DoubleCtrl(self.predictions.toggle)         # double-tap Ctrl
            hotkeys[3] = (0x2 | 0x1, ord("S"), lambda: self.q.put(self.open_settings))  # Ctrl+Alt+S
            threading.Thread(target=windows_hotkey_loop, args=(hotkeys,), daemon=True).start()
        else:
            keyboard.GlobalHotKeys({HOTKEY: self.on_hotkey}).start()
        print(f"Careter running ({MODEL if has_ai_key() else 'no AI key yet'}). Press {HOTKEY} anywhere. "
              "Ctrl+C here to quit.")
        if not WELCOMED:
            self.root.after(1500, lambda: ShortcutsCard(self, first=True))
        if self.predictions:
            print(f"Word prediction: {'on' if PREDICT else 'off'} ({PREDICT_HOTKEY} toggles). "
                  "Tab/Enter take the suggestion, up/down choose, 1-5 pick directly. Settings: Ctrl+Alt+S.")

    def listen_commands(self, lock):
        """The single-instance lock socket also takes commands from `caret_ai.py --settings`."""
        lock.listen(2)
        while True:
            try:
                conn, _ = lock.accept()
                with conn:
                    conn.settimeout(1)
                    if conn.recv(64).strip() == b"CARETER SETTINGS":
                        self.q.put(self.open_settings)
            except OSError:
                return

    def open_settings(self):
        if getattr(self, "settings_win", None) and self.settings_win.alive():
            self.settings_win.focus()
        else:
            self.settings_win = SettingsWindow(self)

    def pump(self):
        while not self.q.empty():
            try:
                self.q.get_nowait()()
            except tk.TclError:
                pass          # window was closed
        self.root.after(30, self.pump)

    def on_hotkey(self):
        # one bubble at a time: ignore repeats / key-repeat / our own synthetic keys
        now = time.time()
        if self.triggering or now - self.last_trigger < 0.8:
            return
        self.triggering, self.last_trigger = True, now
        if self.predictions:
            self.predictions.reset()                     # hide the word strip under the bubble
        try:
            if self.session and self.session.busy:
                self.q.put(self.session.grab_focus)      # working: just bring it forward
                return
            wait_for_hotkey_release()
            if self.session:                             # replace the old bubble (one at a time)
                done = threading.Event()
                self.q.put(lambda: (self.session and self.session.close(), done.set()))
                done.wait(1)
                time.sleep(0.15)
            hwnd, proc = foreground_window()
            caret = caret_rect()
            mouse = pyautogui.position()
            if caret:
                anchor, below_y = (caret[0], caret[1]), caret[2]
            else:
                anchor, below_y = (mouse[0], mouse[1]), mouse[1] + 16
            bounds = monitor_bounds(*anchor)
            self.log_start = len(LOG_LINES)
            log(f"hotkey in app={proc} hwnd={hwnd}")
            selection = ""
            try:
                selection = grab_selection(proc)          # Ctrl+C; skipped in terminals
            except Exception:
                log("selection failed:\n" + traceback.format_exc())
            ctx = ("", selection, "") if selection else read_caret_context(proc)
            log(f"context = {format_context(ctx)[:300] if ctx else 'none'!r}")
            shot = capture(anchor) if USE_SCREENSHOTS else None
            self.q.put(lambda: self.open(anchor, below_y, bounds, shot, hwnd, proc, selection, ctx))
        except Exception:
            log("hotkey failed:\n" + traceback.format_exc())
        finally:
            self.triggering = False

    def open(self, *args):
        self.session = Session(self, *args)

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass


class DoubleCtrl:
    """Calls back when Ctrl is tapped twice quickly: press-release, press-release, with no other key
    in between. Holding Ctrl for a shortcut (Ctrl+C, Ctrl+V…) never counts, and neither do keys that
    Careter itself types."""
    TAP_MAX = 0.30      # seconds Ctrl may be held for one tap
    GAP_MAX = 0.40      # max seconds between the two taps

    def __init__(self, callback):
        self.callback = callback
        self.down_at, self.clean, self.last_tap = None, False, 0.0
        kw = {}
        if IS_WIN:      # skip injected keys (LLKHF_INJECTED), e.g. our own Ctrl+C / Ctrl+V
            kw["win32_event_filter"] = lambda msg, data: not (data.flags & 0x10)
        self.listener = keyboard.Listener(on_press=self.press, on_release=self.release, **kw)
        self.listener.start()

    @staticmethod
    def is_ctrl(key):
        K = keyboard.Key
        return key in (K.ctrl, K.ctrl_l, K.ctrl_r)

    def press(self, key):
        if self.is_ctrl(key):
            if self.down_at is None:            # ignore auto-repeat while held
                self.down_at, self.clean = time.time(), True
        else:                                   # any other key: it's a shortcut, not a tap
            self.clean, self.last_tap = False, 0.0

    def release(self, key):
        if not self.is_ctrl(key) or self.down_at is None:
            return
        now = time.time()
        tap = self.clean and now - self.down_at <= self.TAP_MAX
        self.down_at = None
        if not tap:
            self.last_tap = 0.0
        elif now - self.last_tap <= self.GAP_MAX:
            self.last_tap = 0.0
            threading.Thread(target=self.callback, daemon=True).start()
        else:
            self.last_tap = now


def windows_hotkey_loop(hotkeys):
    """Register hotkeys with Windows itself (RegisterHotKey) and wait for them.
    hotkeys = {id: (modifiers, virtual key, callback)}, e.g. Ctrl+Alt+Space.
    Unlike a keyboard hook this can't get 'stuck' thinking Ctrl/Alt are held."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    MOD_NOREPEAT, WM_HOTKEY = 0x4000, 0x0312
    active = {}
    for hid, (mods, vk, callback) in hotkeys.items():
        if user32.RegisterHotKey(None, hid, mods | MOD_NOREPEAT, vk):
            active[hid] = callback
        else:
            msg = f"Could not register hotkey #{hid}: another app is using it."
            print(msg)
            log(msg)
    if not active:
        return
    msg = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        if msg.message == WM_HOTKEY and msg.wParam in active:
            threading.Thread(target=active[msg.wParam], daemon=True).start()


def single_instance_lock():
    """Exit if Careter is already running (e.g. started at login and again by hand)."""
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if IS_WIN and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        lock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        lock.bind(("127.0.0.1", 47821))
    except OSError:
        print("Careter is already running.")
        sys.exit(0)
    return lock


TASK_NAME = "Careter"


def _task_xml(with_logon):
    """A Windows scheduled task that keeps Careter running: start at login, and every 5 minutes start
    it again if it isn't running (a second copy exits at once: see single_instance_lock)."""
    from xml.sax.saxutils import escape
    import datetime
    here = os.path.dirname(os.path.abspath(__file__))
    pyw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    if not os.path.exists(pyw):
        pyw = sys.executable
    user = f"{os.environ.get('USERDOMAIN', '')}\\{os.environ.get('USERNAME', '')}".lstrip("\\")
    start = datetime.datetime.now().replace(microsecond=0).isoformat()
    logon = (f"<LogonTrigger><Enabled>true</Enabled><UserId>{escape(user)}</UserId></LogonTrigger>"
             if with_logon else "")
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>Keeps Careter running (starts it at login and restarts it if it stops).</Description></RegistrationInfo>
  <Triggers>
    {logon}
    <TimeTrigger>
      <Repetition><Interval>PT5M</Interval><StopAtDurationEnd>false</StopAtDurationEnd></Repetition>
      <StartBoundary>{start}</StartBoundary><Enabled>true</Enabled>
    </TimeTrigger>
  </Triggers>
  <Principals><Principal id="Author"><UserId>{escape(user)}</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <StartWhenAvailable>true</StartWhenAvailable>
    <AllowHardTerminate>true</AllowHardTerminate>
    <IdleSettings><StopOnIdleEnd>false</StopOnIdleEnd><RestartOnIdle>false</RestartOnIdle></IdleSettings>
    <RestartOnFailure><Interval>PT1M</Interval><Count>10</Count></RestartOnFailure>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec><Command>{escape(pyw)}</Command><Arguments>"{escape(os.path.join(here, "caret_ai.py"))}"</Arguments><WorkingDirectory>{escape(here)}</WorkingDirectory></Exec>
  </Actions>
</Task>
"""


def _startup_shortcut():
    return os.path.join(os.environ.get("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs\Startup",
                        "Careter.lnk")


def autostart_command(mode):
    """--autostart on|off|status (Windows)."""
    import subprocess
    import tempfile
    if not IS_WIN:
        print("Auto-start is set up by install.sh on Mac/Linux.")
        return
    run = lambda *a: subprocess.run(["schtasks", *a], capture_output=True, text=True)
    if mode == "status":
        r = run("/Query", "/TN", TASK_NAME)
        print("Auto-start: ON (scheduled task 'Careter')" if r.returncode == 0 else "Auto-start: OFF")
        return
    if mode == "off":
        r = run("/Delete", "/TN", TASK_NAME, "/F")
        lnk = _startup_shortcut()
        if os.path.exists(lnk):
            os.remove(lnk)
        print("Auto-start is OFF. Careter keeps running until you stop it "
              "(python caret_ai.py --stop) or sign out.")
        return
    for with_logon in (True, False):            # some PCs don't let normal users add login triggers
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-16") as f:
            f.write(_task_xml(with_logon))
        r = run("/Create", "/TN", TASK_NAME, "/XML", f.name, "/F")
        os.remove(f.name)
        if r.returncode == 0:
            break
    if r.returncode != 0:
        print("Could not create the scheduled task:", (r.stderr or r.stdout).strip())
        return
    if with_logon and os.path.exists(_startup_shortcut()):
        os.remove(_startup_shortcut())          # the task starts it at login now
    run("/Run", "/TN", TASK_NAME)               # start it now if it isn't running
    print("Auto-start is ON: Careter starts when you sign in and comes back within 5 minutes "
          "if it stops." + ("" if with_logon else " (Login start uses your Startup folder shortcut.)"))


def stop_command():
    """--stop: close the running Careter (with auto-start on, it comes back within 5 minutes)."""
    import subprocess
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and "
          f"$_.CommandLine -like '*caret_ai.py*' -and $_.ProcessId -ne {os.getpid()} -and "
          "$_.CommandLine -notlike '*--stop*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $_.ProcessId }")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    stopped = [x for x in r.stdout.split() if x.isdigit()]
    print(f"Stopped Careter ({len(stopped)} process{'es' if len(stopped) != 1 else ''})." if stopped
          else "Careter wasn't running.")


def learned_data_command(args):
    """--export-learned FILE: write what Careter learned (decrypted) as readable JSON.
    --forget-learned: delete everything it learned (words, word pairs, words you told it to keep)."""
    import json
    from predict import UserModel
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngrams")
    model = os.path.join(folder, "user_model.dat")
    if args[0] == "--export-learned":
        out = args[1] if len(args) > 1 else "careter_learned.json"
        data = UserModel(model, legacy=os.path.join(folder, "user_model.json")).export()
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"Wrote {len(data['words'])} words and {sum(map(len, data['pairs'].values()))} "
              f"word pairs to {os.path.abspath(out)} (plain text: delete it when you're done).")
    else:
        gone = [n for n in ("user_model.dat", "user_model.json", "user_words.dat", "user_words.txt")
                if os.path.exists(os.path.join(folder, n))]
        for n in gone:
            os.remove(os.path.join(folder, n))
        print("Forgot everything Careter learned." if gone else "Nothing learned yet.",
              "Restart Careter if it's running.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--export-learned", "--forget-learned"):
        learned_data_command(sys.argv[1:])
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--autostart":
        autostart_command(sys.argv[2] if len(sys.argv) > 2 else "on")
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--stop":
        stop_command()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--settings":
        try:                                   # open the window in the running Careter
            with socket.create_connection(("127.0.0.1", 47821), timeout=2) as c:
                c.sendall(b"CARETER SETTINGS\n")
        except OSError:
            print("Careter isn't running. Start it first.")
        sys.exit(0)
    _lock = single_instance_lock()
    app = App()
    threading.Thread(target=app.listen_commands, args=(_lock,), daemon=True).start()
    app.run()
