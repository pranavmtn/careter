"""Linux (X11) pieces for Careter's word prediction.

- active_window(): the app you're in (window id, program name, process id)
- KeyGrabber: takes Tab / Enter / up / down / 1-5 for Careter only while the suggestion list shows
  (X11 can't hide single keys from a keyboard hook the way Windows can)
- AtspiReader: where the cursor is and the text before it, through accessibility (AT-SPI), which GNOME
  apps, LibreOffice, Firefox and Qt apps support (Chrome: use the browser helper extension)
- line_colors(): screen pixels, for the grey text's background

Needs an X11 session ("Ubuntu on Xorg"); Wayland doesn't let apps watch keys or place windows freely.
Nothing here logs or keeps what you type.
"""
import os
import queue
import select
import threading
import time


def session_ok():
    """(ok, why): word prediction needs an X11 display."""
    if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" or (
            os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("DISPLAY")):
        return False, "Wayland session: log in with \"Ubuntu on Xorg\" (gear icon on the login screen)"
    if not os.environ.get("DISPLAY"):
        return False, "no X display (DISPLAY is not set)"
    try:
        import Xlib.display  # noqa: F401  (comes with pynput on Linux)
    except ImportError:
        return False, "python-xlib is missing: run .venv/bin/pip install python-xlib"
    return True, ""


# ---------------------------------------------------------------- the active window
_disp, _lock, _atoms = None, threading.Lock(), {}


def _display():
    global _disp
    if _disp is None:
        from Xlib import display
        _disp = display.Display()
    return _disp


def _atom(d, name):
    if name not in _atoms:
        _atoms[name] = d.intern_atom(name)
    return _atoms[name]


def program_name(pid):
    """'gnome-terminal-server', 'chrome', 'gedit'… (the program file, else the short process name)."""
    if not pid:
        return ""
    try:
        return os.path.basename(os.readlink(f"/proc/{pid}/exe")).lower()
    except OSError:
        pass
    try:
        with open(f"/proc/{pid}/comm") as f:
            return f.read().strip().lower()
    except OSError:
        return ""


def active_window():
    """(window id, program name, pid) of the focused app window, or (None, "", 0)."""
    try:
        with _lock:
            d = _display()
            root = d.screen().root
            prop = root.get_full_property(_atom(d, "_NET_ACTIVE_WINDOW"), 0)
            wid = int(prop.value[0]) if prop is not None and len(prop.value) else 0
            if not wid:
                return None, "", 0
            win = d.create_resource_object("window", wid)
            p = win.get_full_property(_atom(d, "_NET_WM_PID"), 0)
            pid = int(p.value[0]) if p is not None and len(p.value) else 0
    except Exception:
        global _disp
        _disp = None                              # a broken connection: open a new one next time
        return None, "", 0
    return wid, program_name(pid), pid


def window_pid(wid):
    try:
        with _lock:
            d = _display()
            p = d.create_resource_object("window", int(wid)).get_full_property(_atom(d, "_NET_WM_PID"), 0)
            return int(p.value[0]) if p is not None and len(p.value) else 0
    except Exception:
        return 0


# ---------------------------------------------------------------- keys for the list
class KeyGrabber(threading.Thread):
    """Grabs a few keys on the whole screen while the suggestion list shows, so the app never gets
    them (like the Windows hook does). on_key(name) gets "tab", "enter", "up", "down" or "1".."5" and
    returns False if it didn't use the key; then the key is let go and typed into the app after all."""

    KEYSYMS = {"tab": ["Tab"], "enter": ["Return", "KP_Enter"], "up": ["Up"], "down": ["Down"],
               **{str(i): [str(i), f"KP_{i}"] for i in range(1, 10)}}

    def __init__(self, on_key, log=print):
        super().__init__(daemon=True)
        self.on_key, self.log = on_key, log
        self.jobs = queue.Queue()
        self.grabbed = set()            # names grabbed right now
        self.last = None                # (name, time) of the last grabbed key press
        self.codes = {}                 # keycode -> name
        self.ok = False
        self.free = threading.Event()   # set while nothing is grabbed
        self.free.set()

    # called from any thread
    def set(self, names):
        self.jobs.put(frozenset(names))

    def release(self):
        if self.grabbed:
            self.free.clear()
        self.jobs.put(frozenset())

    def wait_released(self, timeout=0.3):
        if self.ok:
            self.free.wait(timeout)

    def is_grabbed(self, name):
        return name in self.grabbed

    # the grab thread
    def run(self):
        try:
            from Xlib import X, XK, display, error
            self.X, self.error = X, error
            self.d = display.Display()
            self.root = self.d.screen().root
            for name, syms in self.KEYSYMS.items():
                for s in syms:
                    code = self.d.keysym_to_keycode(XK.string_to_keysym(s))
                    if code:
                        self.codes.setdefault(code, name)
            self.ok = True
        except Exception as e:
            self.log(f"predict: key grab unavailable ({e}); Tab/1-5 then also reach the app")
            return
        fd = self.d.fileno()
        while True:
            try:
                while not self.jobs.empty():
                    names = self.jobs.get_nowait()
                    while not self.jobs.empty():          # only the newest request matters
                        names = self.jobs.get_nowait()
                    self._apply(names)
                if self.d.pending_events() or select.select([fd], [], [], 0.03)[0]:
                    while self.d.pending_events():
                        self._event(self.d.next_event())
            except Exception as e:
                self.log(f"predict: key grab error: {e}")
                time.sleep(0.2)

    def _variants(self):
        X = self.X
        lock, num = X.LockMask, X.Mod2Mask                     # Caps Lock / Num Lock on or off
        return (0, lock, num, lock | num)

    def _apply(self, names):
        if names == self.grabbed:
            if not names:
                self.free.set()
            return
        catch = self.error.CatchError()
        for code, name in self.codes.items():
            want, have = name in names, name in self.grabbed
            if want and not have:
                for m in self._variants():
                    self.root.grab_key(code, m, False, self.X.GrabModeAsync, self.X.GrabModeAsync,
                                       onerror=catch)
            elif have and not want:
                for m in self._variants():
                    self.root.ungrab_key(code, m)
        self.d.sync()
        if catch.get_error():
            self.log("predict: another app holds one of Tab/Enter/arrows/1-5; that key still reaches the app")
        self.grabbed = set(names)
        if self.grabbed:
            self.free.clear()
        else:
            self.free.set()

    def _event(self, ev):
        if ev.type != self.X.KeyPress:
            return                                            # releases of grabbed keys: ignore
        name = self.codes.get(ev.detail)
        if not name:
            return
        self.last = (name, time.time())
        used = False
        try:
            used = self.on_key(name)
        except Exception as e:
            self.log(f"predict: key handler failed: {e}")
        if used is False:                                     # not ours after all: give it to the app
            self._apply(frozenset())
            threading.Thread(target=_retype, args=(name,), daemon=True).start()


def _retype(name):
    import pyautogui
    pyautogui.press({"enter": "enter", "tab": "tab", "up": "up", "down": "down"}.get(name, name), _pause=False)


# ---------------------------------------------------------------- the cursor, via accessibility
class AtspiReader:
    """Follows the focused text box through AT-SPI (accessibility events) and answers read(), like
    the Windows UI Automation reader: (text before the cursor, (x, top, bottom) or None, password?)."""

    def __init__(self, log=print):
        self.log = log
        self.A, self.obj, self.ok, self.why = None, None, False, ""
        self.lock = threading.Lock()
        try:
            import gi
            gi.require_version("Atspi", "2.0")
            from gi.repository import Atspi
            self.A = Atspi
        except Exception as e:
            self.why = f"{type(e).__name__}: {e}"
            log("predict: accessibility (AT-SPI) unavailable, using typed keys and the mouse "
                "position (install: sudo apt install python3-gi gir1.2-atspi-2.0)")
            return
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        A = self.A
        try:
            A.init()
            self.listener = A.EventListener.new(self._event)
            for ev in ("object:text-caret-moved", "object:state-changed:focused"):
                self.listener.register(ev)
            self.ok = True
            self.log("predict: accessibility (AT-SPI) on: exact cursor position where apps support it")
            A.event_main()
        except Exception as e:
            self.ok, self.why = False, str(e)
            self.log(f"predict: accessibility (AT-SPI) failed: {e}")

    def _event(self, e, *_):
        try:
            if e.type.startswith("object:text-caret-moved") or (e.type.endswith("focused") and e.detail1):
                self.obj = e.source
        except Exception:
            pass

    def _point(self, obj, off):
        A = self.A
        SCREEN = A.CoordType.SCREEN
        for i, after in ((off - 1, True), (off, False)):       # the letter before the cursor, else after
            if i < 0:
                continue
            try:
                r = A.Text.get_character_extents(obj, i, SCREEN)
            except Exception:
                continue
            if r and r.height > 0 and (r.x or r.y):
                x = r.x + r.width if after else r.x
                return int(x), int(r.y), int(r.y + r.height)
        return None

    def _read(self, chars, pid):
        obj, A = self.obj, self.A
        if not obj:
            return None
        try:
            if pid and obj.get_process_id() != pid:
                return None                                   # it's from another window
            if hasattr(obj, "get_text_iface") and obj.get_text_iface() is None:
                return None                                   # not a text box
            off = A.Text.get_caret_offset(obj)
            if off is None or off < 0:
                return None
            password = obj.get_role() == A.Role.PASSWORD_TEXT
            text = "" if password else (A.Text.get_text(obj, max(0, off - chars), off) or "")
            return text, self._point(obj, off), password
        except Exception:
            self.obj = None                                  # the box is gone
            return None

    def read(self, chars=300, timeout=0.25, pid=None):
        if not self.ok:
            return None
        box, done = [None], threading.Event()

        def job():
            with self.lock:
                box[0] = self._read(chars, pid)
            done.set()
        threading.Thread(target=job, daemon=True).start()
        done.wait(timeout)
        return box[0]

    def caret(self, pid=None):
        r = self.read(1, 0.15, pid)
        return r[1] if r else None


# ---------------------------------------------------------------- screen pixels
def line_colors(points):
    """{(r, g, b), …} at these screen points (one small screenshot)."""
    import mss
    xs, ys = [int(p[0]) for p in points], [int(p[1]) for p in points]
    left, top = min(xs), min(ys)
    box = {"left": left, "top": top, "width": max(xs) - left + 1, "height": max(ys) - top + 1}
    with mss.mss() as s:
        img = s.grab(box)
        return {img.pixel(x - left, y - top) for x, y in zip(xs, ys)}
