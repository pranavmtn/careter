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
import tempfile
import threading
import time
import wave
import tkinter as tk
from tkinter import scrolledtext

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

IS_MAC = platform.system() == "Darwin"
pyautogui.FAILSAFE = True   # move mouse to a screen corner to abort typing
client = OpenAI()           # reads OPENAI_API_KEY

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


def run_actions(actions, anchor):
    if FOCUS_CLICK:
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


# ---------------------------------------------------------------- UI
class Session:
    """One popup at one spot. Keeps conversation history so you can give follow-ups."""

    def __init__(self, app, anchor, shot):
        self.app, self.anchor, self.shot = app, anchor, shot
        self.msgs = [{"role": "system", "content": SYSTEM}]
        self.busy = False
        self.recorder = Recorder()
        self.decision = None
        self.decided = threading.Event()

        win = self.win = tk.Toplevel(app.root)
        win.title("Caret AI")
        win.attributes("-topmost", True)
        win.geometry(f"460x320+{anchor[0] + 20}+{anchor[1] + 20}")
        win.bind("<Escape>", lambda e: self.close())

        self.log = scrolledtext.ScrolledText(win, height=10, wrap="word", font=("Helvetica", 11))
        self.log.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self.log.tag_config("you", foreground="#2563eb")
        self.log.tag_config("act", foreground="#555555")
        self.log.tag_config("risky", foreground="#dc2626")
        self.log.configure(state="disabled")

        self.confirm_bar = tk.Frame(win)
        self.confirm_label = tk.Label(self.confirm_bar, text="Run these actions?")
        self.confirm_label.pack(side="left")
        tk.Button(self.confirm_bar, text="Stop", command=lambda: self.decide(False)).pack(side="right")
        tk.Button(self.confirm_bar, text="Run ⏎", command=lambda: self.decide(True)).pack(side="right", padx=4)

        bar = self.input_bar = tk.Frame(win)
        bar.pack(fill="x", padx=8, pady=(0, 8))
        self.entry = tk.Entry(bar, font=("Helvetica", 12))
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda e: self.submit())
        self.mic_btn = tk.Button(bar, text="🎤", width=3, command=self.toggle_mic)
        self.mic_btn.pack(side="left", padx=4)
        tk.Button(bar, text="Go", command=self.submit).pack(side="left")

        self.write("Tell me what to do here (type or 🎤).", "act")
        win.after(50, lambda: (win.focus_force(), self.entry.focus_set()))

    # --- thread-safe helpers (agent thread -> UI thread)
    def ui(self, fn):
        self.app.q.put(fn)

    def write(self, text, tag=None):
        def _w():
            self.log.configure(state="normal")
            self.log.insert("end", text + "\n", tag)
            self.log.see("end")
            self.log.configure(state="disabled")
        self.ui(_w)

    # --- input
    def toggle_mic(self):
        if self.recorder.stream is None:
            self.recorder.start()
            self.mic_btn.configure(text="■")
            self.write("Listening… click ■ to stop.", "act")
        else:
            self.mic_btn.configure(text="…")

            def work():
                try:
                    text = self.recorder.stop_and_transcribe()
                except Exception as e:
                    self.write(f"Mic error: {e}", "risky")
                    text = ""
                self.ui(lambda: self.mic_btn.configure(text="🎤"))
                if text.strip():
                    self.ui(lambda: (self.entry.delete(0, "end"), self.entry.insert(0, text), self.submit()))
            threading.Thread(target=work, daemon=True).start()

    def submit(self):
        goal = self.entry.get().strip()
        if not goal or self.busy:
            return
        self.entry.delete(0, "end")
        self.write(f"You: {goal}", "you")
        self.busy = True
        threading.Thread(target=self.agent, args=(goal,), daemon=True).start()

    # --- confirmation
    def ask_confirm(self, actions, risky):
        self.decided.clear()

        def show():
            self.confirm_label.configure(text="⚠ Risky — run these?" if risky else "Run these actions?",
                                         fg="#dc2626" if risky else "black")
            self.confirm_bar.pack(fill="x", padx=8, pady=(0, 4), before=self.input_bar)
            self.win.bind("<Return>", lambda e: self.decide(True))
            self.win.focus_force()
        self.ui(show)
        self.decided.wait()
        return self.decision

    def decide(self, ok):
        self.decision = ok
        self.confirm_bar.pack_forget()
        self.win.unbind("<Return>")
        self.decided.set()

    # --- the loop: look -> think -> confirm -> act -> look again
    def agent(self, goal):
        try:
            if len(self.msgs) > 1:      # follow-up: take a fresh look first
                self.hide()
                self.shot = capture(self.anchor)
                self.show()
            for step in range(MAX_STEPS):
                strip_old_images(self.msgs)
                text = f"Goal: {goal}" if step == 0 else "Screen after your actions. Continue toward the goal."
                full, crop = self.shot
                self.msgs.append({"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": full}},
                    {"type": "image_url", "image_url": {"url": crop}},
                ]})
                self.write("Thinking…", "act")
                resp = client.chat.completions.create(model=MODEL, messages=self.msgs)
                raw = resp.choices[0].message.content
                self.msgs.append({"role": "assistant", "content": raw})
                reply = parse_reply(raw)

                if reply.get("say"):
                    self.write(f"AI: {reply['say']}")
                actions = reply.get("actions") or []
                if not actions:
                    break
                risky = bool(reply.get("risky"))
                for a in actions:
                    self.write("   • " + describe(a), "risky" if risky else "act")
                if not self.ask_confirm(actions, risky):
                    self.write("Stopped.", "act")
                    break

                self.hide()
                run_actions(actions, self.anchor)
                self.shot = capture(self.anchor)
                self.show()
                if reply.get("done"):
                    break
            else:
                self.write(f"Reached {MAX_STEPS} steps; stopping.", "act")
        except pyautogui.FailSafeException:
            self.show()
            self.write("Aborted (mouse moved to corner).", "risky")
        except Exception as e:
            self.show()
            self.write(f"Error: {e}", "risky")
        finally:
            self.busy = False

    def hide(self):
        done = threading.Event()
        self.ui(lambda: (self.win.withdraw(), done.set()))
        done.wait()
        time.sleep(0.25)

    def show(self):
        self.ui(lambda: (self.win.deiconify(), self.win.attributes("-topmost", True)))

    def close(self):
        if self.recorder.stream is not None:
            self.recorder.stream.stop()
            self.recorder.stream.close()
        self.decision = False
        self.decided.set()
        self.win.destroy()


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.q = queue.Queue()
        self.root.after(30, self.pump)
        keyboard.GlobalHotKeys({HOTKEY: self.on_hotkey}).start()
        print(f"Caret AI running ({MODEL}). Put your pointer somewhere and press {HOTKEY}. Ctrl+C here to quit.")

    def pump(self):
        while not self.q.empty():
            try:
                self.q.get_nowait()()
            except tk.TclError:
                pass          # window was closed
        self.root.after(30, self.pump)

    def on_hotkey(self):
        anchor = pyautogui.position()
        shot = capture(anchor)    # screenshot before the popup appears
        self.q.put(lambda: Session(self, anchor, shot))

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    App().run()
