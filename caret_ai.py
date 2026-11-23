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
CONTEXT_CHARS = 10      # characters read before and after the caret
USE_SCREENSHOTS = False # text-only for now: the AI reads only the text at the caret
DEBUG = True            # show the live log inside the bubble (toggle with 🐞)
SECTION_BREAKS = True   # stop reading at a blank line (treat it as the end of the section)
AUTO_SUGGEST = True     # when the bubble opens, analyse the spot and offer next actions

# Word prediction while you type (Windows). Runs a small local model: no API key, nothing leaves the PC.
PREDICT = os.getenv("CARET_PREDICT", "1") != "0"
PREDICT_ENGINE = os.getenv("CARET_PREDICT_ENGINE", "llm")    # "llm" (smarter) or "ngram" (word counts only)
PREDICT_LLM_TIMEOUT = 0.6     # if the LLM takes longer, show the word-count suggestions instead
PREDICT_MODEL = os.getenv("CARET_PREDICT_MODEL", "HuggingFaceTB/SmolLM2-135M")   # llm engine: a *base* model
PREDICT_DELAY = 0.18          # seconds of no typing before suggestions appear
PREDICT_COUNT = 5             # suggestions shown, picked with keys 1-5
PREDICT_IN_TERMINALS = False  # shell commands aren't prose; off by default
PREDICT_HOTKEY = "double-tap Ctrl (or Ctrl+Alt+P)"  # turns predictions (and auto-correct) on/off
AUTOCORRECT = os.getenv("CARET_AUTOCORRECT", "1") != "0"   # fix clear typos when you press space
LEARN = os.getenv("CARET_LEARN", "1") != "0"   # learn your words & word pairs (ngrams/user_model.json)

IS_MAC = platform.system() == "Darwin"
IS_WIN = platform.system() == "Windows"
# Never send Ctrl+C to these (it would interrupt the running command)
TERMINALS = {"windowsterminal.exe", "wt.exe", "powershell.exe", "pwsh.exe", "cmd.exe",
             "conhost.exe", "mintty.exe", "openconsole.exe", "alacritty.exe", "wezterm-gui.exe"}
pyautogui.FAILSAFE = True   # move mouse to a screen corner to abort typing
client = OpenAI()           # reads OPENAI_API_KEY

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
        ctrl, tp = auto.GetFocusedControl(), None
        if ctrl is None:
            return None
        try:
            if ctrl.IsPassword:
                return "", None, True
        except Exception:
            pass
        for _ in range(6):
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


def trim_context(before, selected, after, chars=CONTEXT_CHARS):
    """Keep N characters each side of the caret, only within the current section."""
    before, after = before.replace("\r\n", "\n"), after.replace("\r\n", "\n")
    if SECTION_BREAKS:
        before = re.split(r"\n[ \t]*\n", before)[-1]
        after = re.split(r"\n[ \t]*\n", after)[0]
    before, after = before[-chars:], after[:chars]
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
        sentinel = f"__careter_{time.time()}__"
        pyperclip.copy(sentinel)
        for _ in range(chars):
            send_hotkey("shift", extend_key)
        time.sleep(0.05)
        send_hotkey(mod, "c")
        time.sleep(0.15)
        text = pyperclip.paste()
        log(f"  shift+{extend_key} x{chars} copied: {'nothing' if text == sentinel else repr(text)}")
        if text != sentinel and text:
            send_hotkey(collapse_key)        # collapse the selection back onto the caret
            time.sleep(0.03)
            return text
        return ""                             # at the start/end: nothing selected, caret didn't move

    try:
        before = copy_side("left", "right")
        after = copy_side("right", "left")
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
STRIP_FONT = ("Segoe UI", 11) if IS_WIN else ("Helvetica", 13)
STRIP_NUM_FONT = ("Segoe UI", 9) if IS_WIN else ("Helvetica", 11)
STRIP_RADIUS = 14


class Session:
    """A small tooltip bubble at the caret/selection. Keeps history so you can ask follow-ups."""

    def __init__(self, app, anchor, below_y, bounds, shot, hwnd=None, proc="", selection="", ctx=None):
        self.app, self.anchor, self.below_y, self.bounds, self.shot = app, anchor, below_y, bounds, shot
        self.hwnd, self.proc, self.selection, self.ctx = hwnd, proc, selection, ctx
        self.msgs = [{"role": "system", "content": SYSTEM}]
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
        win.overrideredirect(True)                 # no title bar: looks like a tooltip
        win.attributes("-topmost", True)
        win.configure(bg=ACCENT)                   # 1px accent border
        win.bind("<Escape>", lambda e: self.escape())
        box = self.box = tk.Frame(win, bg=BG, padx=8, pady=6)
        box.pack(padx=1, pady=1)

        # selection chip
        if selection:
            preview = " ".join(selection.split())
            preview = preview[:60] + ("…" if len(preview) > 60 else "")
            tk.Label(box, text=f"✎ “{preview}”", bg=BG, fg=MUTED, font=FONT, anchor="w",
                     justify="left", wraplength=WRAP).pack(fill="x", pady=(0, 4))

        # what was read around the caret
        if ctx and not selection:
            seen = (ctx[0] + "▮" + ctx[2]).replace("\n", "⏎")
            tk.Label(box, text=f"📄 caret: “{seen}”", bg=BG, fg=MUTED, font=FONT,
                     anchor="w").pack(fill="x", pady=(0, 4))
        elif not ctx and not selection:
            tk.Label(box, text="⚠ couldn't read text at the caret (see careter.log)", bg=BG, fg=RED,
                     font=FONT, anchor="w").pack(fill="x", pady=(0, 4))

        # input row
        row = tk.Frame(box, bg=BG)
        row.pack(fill="x")
        handle = tk.Label(row, text="✦", bg=BG, fg=ACCENT, font=FONT_B, cursor="fleur")
        handle.pack(side="left", padx=(0, 6))
        handle.bind("<ButtonPress-1>", self.drag_start)
        handle.bind("<B1-Motion>", self.drag_move)
        self.entry = tk.Entry(row, bg=FIELD, fg=FG, insertbackground=FG, relief="flat",
                              font=FONT, width=34, highlightthickness=0)
        self.entry.pack(side="left", fill="x", expand=True, ipady=4)
        self.entry.bind("<Return>", lambda e: self.submit())
        self.placeholder = "Ask about the selection…" if selection else "What should I do here?"
        self.set_placeholder()
        self.entry.bind("<FocusIn>", lambda e: self.clear_placeholder())
        self.entry.bind("<Key>", lambda e: self.clear_placeholder())
        self.mic = self.icon(row, "🎤", self.toggle_mic)
        self.icon(row, "🐞", self.toggle_debug)
        self.icon(row, "✕", self.close)

        # body (hidden until there is something to show)
        self.status = tk.Label(box, bg=BG, fg=MUTED, font=FONT, anchor="w")
        self.answer = tk.Label(box, bg=BG, fg=FG, font=FONT, anchor="w", justify="left", wraplength=WRAP)
        self.steps = tk.Label(box, bg=BG, fg=MUTED, font=("Consolas", 9) if IS_WIN else ("Menlo", 11),
                              anchor="w", justify="left", wraplength=WRAP)
        self.confirm = tk.Frame(box, bg=BG)
        self.confirm_label = tk.Label(self.confirm, bg=BG, fg=MUTED, font=FONT)
        self.confirm_label.pack(side="left")
        self.button(self.confirm, "Stop", lambda: self.decide(False), FIELD).pack(side="right")
        self.button(self.confirm, "Run ⏎", lambda: self.decide(True), ACCENT).pack(side="right", padx=4)
        self.summary = tk.Label(box, bg=BG, fg=MUTED, font=FONT, anchor="w", justify="left", wraplength=WRAP)
        self.chips = tk.Frame(box, bg=BG)
        self.tools = tk.Frame(box, bg=BG)
        self.button(self.tools, "Copy", self.copy_answer, FIELD).pack(side="left")
        self.button(self.tools, "Insert", self.insert_answer, FIELD).pack(side="left", padx=4)

        # debug log panel (packed last so it sits at the bottom)
        self.debug = tk.Frame(box, bg=BG)
        self.debug_text = tk.Text(self.debug, height=10, width=60, bg="#11101a", fg="#c4f0c5",
                                  insertbackground=FG, relief="flat", wrap="word",
                                  font=("Consolas", 8) if IS_WIN else ("Menlo", 10))
        self.debug_text.pack(fill="both", expand=True)
        dbar = tk.Frame(self.debug, bg=BG)
        dbar.pack(fill="x", pady=(2, 0))
        self.button(dbar, "Copy log", self.copy_log, FIELD).pack(side="left")
        self.button(dbar, "Clear", lambda: self.debug_text.delete("1.0", "end"), FIELD).pack(side="left", padx=4)
        for line in LOG_LINES[app.log_start:]:      # lines from this hotkey press
            self.debug_text.insert("end", line + "\n")
        self.debug_text.see("end")
        self.on_log = lambda line: self.ui(lambda: (self.debug_text.insert("end", line + "\n"),
                                                    self.debug_text.see("end")))
        LOG_LISTENERS.append(self.on_log)
        self.debug_open = False
        if DEBUG:
            self.toggle_debug()

        self.place()
        win.after(30, self.grab_focus)
        if AUTO_SUGGEST:
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
        lbl = tk.Label(parent, text=text, bg=BG, fg=MUTED, font=FONT, cursor="hand2", padx=4)
        lbl.pack(side="left")
        lbl.bind("<Button-1>", lambda e: cmd())
        return lbl

    def button(self, parent, text, cmd, bg):
        b = tk.Label(parent, text=text, bg=bg, fg=FG, font=FONT, padx=10, pady=2, cursor="hand2")
        b.bind("<Button-1>", lambda e: cmd())
        return b

    def set_placeholder(self):
        self.entry.delete(0, "end")
        self.entry.insert(0, self.placeholder)
        self.entry.configure(fg=MUTED)
        self.has_placeholder = True

    def clear_placeholder(self):
        if self.has_placeholder:
            self.entry.delete(0, "end")
            self.entry.configure(fg=FG)
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

    def set_status(self, text, color=MUTED):
        self.ui(lambda: self.show_part(self.status, text, fg=color))

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
        self.ui(lambda: self.show_part(self.steps, text, fg=RED if risky else MUTED))

    # --- debug panel
    def toggle_debug(self):
        self.debug_open = not self.debug_open
        if self.debug_open:
            self.debug.pack(side="bottom", fill="both", expand=True, pady=(8, 0))
        else:
            self.debug.pack_forget()
        self.place()

    def copy_log(self):
        pyperclip.copy(self.debug_text.get("1.0", "end"))
        self.set_status("Log copied ✓")

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
            chip = self.button(self.chips, text, lambda t=text: self.ask(t), FIELD)
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
            self.recorder.start()
            self.mic.configure(text="■", fg=RED)
            self.set_status("Listening… click ■ to stop")
        else:
            self.mic.configure(text="…", fg=MUTED)

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

    def submit(self):
        goal = "" if self.has_placeholder else self.entry.get().strip()
        if not goal or self.busy:
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
                                         fg=RED if risky else MUTED)
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
        if self.on_log in LOG_LISTENERS:
            LOG_LISTENERS.remove(self.on_log)
        if self.recorder.stream is not None:
            self.recorder.stream.stop()
            self.recorder.stream.close()
        self.decision = False
        self.decided.set()
        self.win.destroy()
        if self.app.session is self:
            self.app.session = None


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
        self.user_model = None
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
            self.predictor = ngram
            personal = lambda engine: engine
            if LEARN:
                from predict import PersonalPredictor, UserModel
                self.user_model = UserModel(os.path.join(folder, "user_model.json"))
                personal = lambda engine: PersonalPredictor(engine, self.user_model)
                self.predictor = personal(ngram)
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
        if not self.user_model or self.buf_proc in TERMINALS:
            return
        m = re.search(r"((?:[^\W\d_]|['’\-])+)\s\Z", self.buf)
        if not m:
            return
        word, before = m.group(1), self.buf[: m.start()]
        if not before.strip() or not before[-1].isspace():
            return                             # first word in a box (maybe a username/password)
        tail = before.rstrip()
        pm = re.search(r"((?:[^\W\d_]|['’\-])+)\Z", tail)
        prev = pm.group(1).lower() if (pm and tail[-1] not in ".!?") else None
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
        if down and vk in (0x20, 0x08) and self.enabled and self.speller and not self.fixing:
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
            if vk == 0x09 and not (ctrl or alt or shift or win):          # Tab
                pick = 0
            elif not (ctrl or alt or shift or win):
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
        hwnd = self.user32.GetForegroundWindow()
        if hwnd != self.buf_hwnd:              # another window: what we typed before doesn't apply
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
        if self.buf_proc in TERMINALS:
            return None
        m = re.search(r"([^\W\d_]+)\Z", self.buf)
        if not m:
            return None
        word, before = m.group(1), self.buf[: m.start()]
        # Only a word that follows another word in this box: never the first thing typed in a field
        # (that protects passwords and usernames), and never glued to symbols like "x.teh" or "@teh".
        if not before.strip() or not before[-1].isspace():
            return None
        tail = before.rstrip()
        pm = re.search(r"([^\W\d_]+)\Z", tail)
        prev = pm.group(1).lower() if (pm and tail[-1] not in ".!?") else None
        new = self.speller.correct(word, prev)
        if not new or new == word:
            return None
        return lambda: self.apply_fix(word, new)

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
        if self.visible and l <= x < r and t <= y < b:
            return                             # a click on our own strip
        self.reset()                           # the caret probably moved

    def reset(self):
        self.buf, self.last_fix = "", None
        self.gen += 1
        if self.timer:
            self.timer.cancel()
        self.hide()

    def schedule(self):
        self.gen += 1
        self.hide()
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(PREDICT_DELAY, self.run, args=(self.gen,))
        self.timer.daemon = True
        self.timer.start()

    # --- predict
    def run(self, gen):
        """Never logs what you typed: only why a suggestion was or wasn't shown."""
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
            uia = run_with_timeout(lambda: read_before_caret(300), 0.4)
            if uia and uia[2]:
                return why("skipped, password box")
            source = "uia" if uia and uia[0].strip() else "keys"
            text = uia[0] if source == "uia" else self.buf
            if gen != self.gen:
                return
            if not text.strip():
                return why(f"no text before caret in {proc} (uia={'yes' if uia else 'no'})")
            t0 = time.time()
            words = self.predictor.predict(text, PREDICT_COUNT, cancelled=lambda: gen != self.gen)
            ms = (time.time() - t0) * 1000
            if gen != self.gen:
                return
            if not words:
                return why(f"no suggestions ({source}, {len(text)} chars, ends {'space' if text[-1] == ' ' else 'mid-word' if text[-1].isalnum() else 'punct'}, {ms:.0f}ms)")
            where = "uia"
            point = uia[1] if uia else None
            if not point:
                point, where = caret_rect(), "caret"
            if not point:
                mx, my = pyautogui.position()
                point, where = (mx, my, my + 16), "mouse"
            engine = getattr(self.predictor, "last", "ngram")
            why(f"{len(words)} words by {engine} in {ms:.0f}ms from {source} in {proc}, strip at {where} {point[0]},{point[2]}")
            self.app.q.put(lambda: self.show(words, point, gen))
        except Exception:
            log("predict failed:\n" + traceback.format_exc())

    def is_ours(self, hwnd):
        from ctypes import wintypes, byref
        pid = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(hwnd, byref(pid))
        return pid.value == os.getpid()

    def accept(self, choice):
        word, insert = choice
        self.gen += 1
        self.hide()
        time.sleep(0.03)
        try:
            if insert.isascii():
                pyautogui.write(insert, interval=0.004)
            else:
                paste(insert)
        except Exception:
            log("predict: insert failed:\n" + traceback.format_exc())
            return
        self.buf = (self.buf + insert)[-self.BUF_MAX:]
        self.learn_last()
        self.schedule()                        # offer the word after it straight away

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
        self.strip_box = tk.Frame(win, bg=STRIP_BG, padx=12, pady=6)
        self.strip_box.pack()
        win.geometry("+-2000+-2000")
        win.update_idletasks()
        self.strip_hwnd = w.GetAncestor(win.winfo_id(), 2)          # GA_ROOT
        GWL_EXSTYLE, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW = -20, 0x08000000, 0x80
        style = w.GetWindowLongW(self.strip_hwnd, GWL_EXSTYLE)
        w.SetWindowLongW(self.strip_hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
        try:
            win.attributes("-disabled", True)       # a disabled window can't take keyboard focus
        except tk.TclError:
            pass
        w.ShowWindow(self.strip_hwnd, 0)            # SW_HIDE
        self.strip_checked = False
        self.round_native = self.round_corners()

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
        for child in self.strip_box.winfo_children():
            child.destroy()
        for i, (label, hint, cmd) in enumerate(items):
            cell = tk.Frame(self.strip_box, bg=STRIP_BG, cursor="hand2" if cmd else "")
            cell.pack(side="left", padx=(0 if i == 0 else 14, 0))
            parts = []
            if hint:                                   # the key that picks it, before the word
                parts.append(tk.Label(cell, text=hint, bg=STRIP_BG, fg=STRIP_NUM, font=STRIP_NUM_FONT))
            weight = (STRIP_FONT[0], STRIP_FONT[1], "bold") if (i == 0 and hint) else STRIP_FONT
            parts.append(tk.Label(cell, text=label, bg=STRIP_BG, fg=STRIP_FG, font=weight, padx=3))
            for w in parts:
                w.pack(side="left")
            if cmd:
                for w in [cell] + parts:
                    w.bind("<Button-1>", lambda e, c=cmd: c())

    def place_strip(self, point):
        x, top, bottom = point
        self.strip.update_idletasks()
        w_, h = self.strip.winfo_reqwidth(), self.strip.winfo_reqheight()
        left, mtop, right, mbottom = monitor_bounds(x, top)
        y = bottom + 4 if bottom + 4 + h < mbottom else top - h - 4
        x = max(left, min(x, right - w_))
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

    def show(self, words, point, gen):
        if gen != self.gen or not self.strip:
            return
        try:
            self._show(words, point)
        except Exception:
            log("predict: strip failed:\n" + traceback.format_exc())

    def _show(self, words, point):
        items = [(w, str(i + 1),
                  lambda c=(w, ins): threading.Thread(target=self.accept, args=(c,), daemon=True).start())
                 for i, (w, ins) in enumerate(words)]
        self.fill(items)
        self.shown, self.visible = list(words), True
        self.place_strip(point)

    def flash(self, text, point):
        if not self.strip:
            return
        self.fill([(text, "", None)])
        self.shown, self.visible = [], True
        self.place_strip(point)
        g = self.gen
        self.strip.after(1200, lambda: self.gen == g and self.hide())

    def hide(self):
        was = self.visible
        self.visible, self.shown = False, []
        if was and self.strip:
            self.app.q.put(lambda: self.w.ShowWindow(self.strip_hwnd, 0))


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.q = queue.Queue()
        self.session = None
        self.triggering = False
        self.last_trigger = 0.0
        self.log_start = 0
        self.root.after(30, self.pump)
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
            threading.Thread(target=windows_hotkey_loop, args=(hotkeys,), daemon=True).start()
        else:
            keyboard.GlobalHotKeys({HOTKEY: self.on_hotkey}).start()
        print(f"Careter running ({MODEL}). Press {HOTKEY} anywhere. Ctrl+C here to quit.")
        if self.predictions:
            print(f"Word prediction: {'on' if PREDICT else 'off'} ({PREDICT_HOTKEY} toggles). "
                  "Keys 1-5 (or Tab for the first) pick a word.")

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


if __name__ == "__main__":
    _lock = single_instance_lock()
    App().run()
