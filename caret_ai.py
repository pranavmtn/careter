"""
Caret AI - an assistant that lives at your pointer.

Put your mouse/caret anywhere (terminal, ebook, browser, editor), press the
hotkey, then type or speak what you want. It screenshots the screen with a
marker at that spot, reads the text around it, proposes actions, asks you to
confirm, runs them, looks at the result, and keeps going until done.

Hotkey:  Ctrl+Alt+Space   (change HOTKEY below)
Abort:   Esc in the popup, or slam the mouse into a screen corner while it types.
"""
import base64
import io
import json
import os
import platform
import queue
import re
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
from pynput import keyboard

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
SECTION_BREAKS = True   # stop reading at a blank line (treat it as the end of the section)
AUTO_SUGGEST = True     # when the bubble opens, analyse the spot and offer next actions

IS_MAC = platform.system() == "Darwin"
IS_WIN = platform.system() == "Windows"
# Never send Ctrl+C to these (it would interrupt the running command)
TERMINALS = {"windowsterminal.exe", "wt.exe", "powershell.exe", "pwsh.exe", "cmd.exe",
             "conhost.exe", "mintty.exe", "openconsole.exe", "alacritty.exe", "wezterm-gui.exe"}
pyautogui.FAILSAFE = True   # move mouse to a screen corner to abort typing
client = OpenAI()           # reads OPENAI_API_KEY

SUGGEST_SYSTEM = """You help a user who just called an assistant at their text caret.
You get the text around the caret ("▮" marks the caret, [[ ]] marks a selection; this is only the
section the caret is in) and/or a zoomed screenshot of that spot.
1. In one short line, say what this is (e.g. "Bullet list of essay topics", "PowerShell after a failed pip install").
2. Offer 2-4 next actions the user most likely wants right here, as button labels of at most 5 words
   (e.g. "Fix grammar", "Continue writing", "Explain this", "Fix the error").
Reply ONLY with JSON: {"summary": "...", "suggestions": ["...", "..."]}"""

SYSTEM = f"""You are Caret AI, an assistant that lives at the user's pointer on their {platform.system()} desktop.
Each turn you receive the user's goal, a screenshot with a RED CROSSHAIR at the spot the user chose,
and a zoomed crop around that spot. Read the text near the crosshair carefully: that is the user's
context (a terminal prompt and its output, a paragraph in an ebook, a form field, code, ...).

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
- Terminal: type the command, press enter, then wait. Next turn you will see the output; read it and
  continue (fix errors, run the next step) until the goal is met.
- If the user only wants information (explain, translate, summarize, define what is on screen), put the
  answer in "say", leave "actions" empty and set "done": true.
- Set "risky": true for anything destructive or hard to undo (deleting files, sudo, force push, killing
  processes, installing/uninstalling, sending messages, payments).
- Set "done": true when the goal is achieved or you cannot proceed, and say why.
- TEXT AROUND CARET (if given) is the section the user is working in: "▮" marks the caret and
  [[ ]] marks the selection. Prefer it over reading the screenshot; use the screenshot for layout.
- If SELECTED TEXT is given, that is what the user means by "this", "the text", "selected". To change it,
  return ONE "type" action containing the full replacement text: typing replaces the selection. No enter.
- If the crosshair is on empty space, use the whole screenshot (the active window) to find the context.
- Never pretend to see output that is not on screen. Keep each turn's action list short.
"""


# ---------------------------------------------------------------- screen
def capture(anchor):
    """Screenshot the monitor under the pointer, mark the pointer, return (full, crop) as JPEG data URLs."""
    x, y = anchor
    with mss.mss() as sct:
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


def paste(text):
    old = pyperclip.paste()
    pyperclip.copy(text)
    pyautogui.hotkey("command" if IS_MAC else "ctrl", "v")
    time.sleep(0.2)
    pyperclip.copy(old)


def wait_for_hotkey_release(timeout=1.5):
    """Don't send keys while the user still holds Ctrl+Alt+Space."""
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


def log(msg):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")
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
        with pyautogui.hold("shift"):
            pyautogui.press(extend_key, presses=chars, interval=0.005)
        time.sleep(0.05)
        pyautogui.hotkey(mod, "c")
        time.sleep(0.15)
        text = pyperclip.paste()
        if text != sentinel and text:
            pyautogui.press(collapse_key)    # collapse the selection back onto the caret
            time.sleep(0.03)
            return text
        return ""                             # at the start/end: nothing selected, caret didn't move

    try:
        before = copy_side("left", "right")
        after = copy_side("right", "left")
    finally:
        pyperclip.copy(old)
    return trim_context(before, "", after)


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
    with mss.mss() as sct:
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
    pyautogui.hotkey("command" if IS_MAC else "ctrl", "c")
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
                pyautogui.hotkey(*keys)
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
            tk.Label(box, text="📷 couldn't read text here, using the screenshot", bg=BG, fg=MUTED,
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

    # --- analyse the spot and offer next actions
    def suggest(self):
        content = []
        if self.ctx:
            content.append({"type": "text", "text": "TEXT AROUND CARET:\n" + format_context(self.ctx)})
        content.append({"type": "image_url", "image_url": {"url": self.shot[1], "detail": "high"}})
        if not self.ctx:   # no text from the app: give the whole screen too
            content.append({"type": "image_url", "image_url": {"url": self.shot[0], "detail": "high"}})
        try:
            resp = client.chat.completions.create(model=MODEL, messages=[
                {"role": "system", "content": SUGGEST_SYSTEM}, {"role": "user", "content": content}])
            data = parse_reply(resp.choices[0].message.content)
        except Exception as e:
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
                self.shot = capture(self.anchor)
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
                    text = "Screen after your actions. Continue toward the goal."
                full, crop = self.shot
                self.msgs.append({"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": full, "detail": "high"}},
                    {"type": "image_url", "image_url": {"url": crop, "detail": "high"}},
                ]})
                self.set_status("Thinking…")
                resp = client.chat.completions.create(model=MODEL, messages=self.msgs)
                raw = resp.choices[0].message.content
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
                run_actions(actions, self.anchor,
                            click_to_focus=FOCUS_CLICK and not self.hwnd and not self.selection)
                self.selection = ""      # it has been replaced/used
                self.ctx = None
                self.shot = capture(self.anchor)
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
            self.show()
            self.set_status(f"Error: {e}", RED)
        finally:
            self.busy = False

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
        if self.recorder.stream is not None:
            self.recorder.stream.stop()
            self.recorder.stream.close()
        self.decision = False
        self.decided.set()
        self.win.destroy()
        if self.app.session is self:
            self.app.session = None


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.q = queue.Queue()
        self.session = None
        self.triggering = False
        self.last_trigger = 0.0
        self.root.after(30, self.pump)
        if IS_WIN:
            # OS-level hotkey: exact Ctrl+Alt+Space only, never fires from normal typing or clicks
            threading.Thread(target=windows_hotkey_loop, args=(self.on_hotkey,), daemon=True).start()
        else:
            keyboard.GlobalHotKeys({HOTKEY: self.on_hotkey}).start()
        print(f"Careter running ({MODEL}). Press {HOTKEY} anywhere. Ctrl+C here to quit.")

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
            ctx = run_with_timeout(read_text_context, 1.5)
            # UI Automation gives the selection directly; only fall back to Ctrl+C when it can't
            selection = ctx[1] if ctx else grab_selection(proc)
            if not ctx and not selection:
                ctx = read_context_by_keys(proc)
            elif not ctx and selection:
                ctx = ("", selection, "")
            log(f"app={proc} context={'none' if not ctx else format_context(ctx)[:300]!r}")
            shot = capture(anchor)
            self.q.put(lambda: self.open(anchor, below_y, bounds, shot, hwnd, proc, selection, ctx))
        finally:
            self.triggering = False

    def open(self, *args):
        self.session = Session(self, *args)

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass


def windows_hotkey_loop(callback):
    """Register Ctrl+Alt+Space with Windows itself (RegisterHotKey) and wait for it.
    Unlike a keyboard hook this can't get 'stuck' thinking Ctrl/Alt are held."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    MOD_ALT, MOD_CONTROL, MOD_NOREPEAT, VK_SPACE, WM_HOTKEY = 0x1, 0x2, 0x4000, 0x20, 0x0312
    if not user32.RegisterHotKey(None, 1, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, VK_SPACE):
        print("Could not register Ctrl+Alt+Space: another app is using it.")
        return
    msg = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        if msg.message == WM_HOTKEY:
            threading.Thread(target=callback, daemon=True).start()


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
