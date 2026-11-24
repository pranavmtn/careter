// Careter for Chrome: word suggestions under the caret, keys 1-5 / Tab to pick, auto-correct on space
// (Backspace right after undoes it), learning from what you type, double-tap Ctrl to turn it off/on.
// Never active in password, email, phone, number or card/login-looking fields.
(() => {
  if (window.__careterLoaded) return;
  window.__careterLoaded = true;

  const MAX = 300, DELAY = 50, COUNT = 5;
  let state = { enabled: true, desktop: false, checked: 0 };
  let timer = null, shown = [], strip = null, box = null, lastFix = null;
  let sel = 0, ghost = null, ghostFor = null;
  let dragOffset = null, dragging = null, lastDefault = null;   // you can drag the list out of the way
  let pending = false, queued = "";                  // keys typed while a correction is being looked up
  let gen = 0;

  const ask = (msg) => new Promise((resolve) => {
    try {
      chrome.runtime.sendMessage({ careter: true, ...msg }, (r) => resolve(chrome.runtime.lastError ? {} : r || {}));
    } catch (e) { resolve({}); }                     // extension reloaded: this page's copy is stale
  });

  async function refreshState(force) {
    if (!force && Date.now() - state.checked < 15000) return;
    const r = await ask({ type: "state" });
    if ("enabled" in r) state = { ...r, checked: Date.now() };
  }
  const active = () => state.enabled && !state.desktop;

  // ---------------------------------------------------------------- which boxes
  const PRIVATE = /pass|pwd|user|login|email|e-mail|card|cvv|cvc|otp|pin|iban|ssn|phone|tel/i;
  function target() {
    let el = document.activeElement;
    while (el && el.shadowRoot && el.shadowRoot.activeElement) el = el.shadowRoot.activeElement;
    if (!el) return null;
    if (el.tagName === "TEXTAREA") return el;
    if (el.tagName === "INPUT") {
      const t = (el.type || "text").toLowerCase();
      if (t !== "text" && t !== "search" && t !== "") return null;
      const hint = `${el.name} ${el.id} ${el.autocomplete} ${el.getAttribute("aria-label") || ""} ${el.placeholder || ""}`;
      return PRIVATE.test(hint) ? null : el;
    }
    return el.isContentEditable ? el : null;
  }
  const isField = (el) => el.tagName === "INPUT" || el.tagName === "TEXTAREA";

  // ---------------------------------------------------------------- text and caret
  function textBefore(el) {
    if (isField(el)) {
      if (el.selectionStart == null || el.selectionStart !== el.selectionEnd) return null;
      return el.value.slice(Math.max(0, el.selectionStart - MAX), el.selectionStart);
    }
    const sel = getSelection();
    if (!sel || !sel.rangeCount || !sel.isCollapsed) return null;
    const r = sel.getRangeAt(0);
    if (!el.contains(r.startContainer)) return null;
    const before = document.createRange();
    before.selectNodeContents(el);
    before.setEnd(r.startContainer, r.startOffset);
    return before.toString().slice(-MAX).replace(/\u00a0/g, " ");
  }

  function textAfter(el) {
    if (isField(el)) return el.value.slice(el.selectionEnd, el.selectionEnd + 40);
    const s2 = getSelection();
    if (!s2 || !s2.rangeCount) return "x";
    const r = s2.getRangeAt(0), after = document.createRange();
    after.selectNodeContents(el);
    after.setStart(r.endContainer, r.endOffset);
    return after.toString().slice(0, 40);
  }

  const MIRROR = ["boxSizing", "width", "height", "overflowX", "overflowY", "borderTopWidth", "borderRightWidth",
    "borderBottomWidth", "borderLeftWidth", "paddingTop", "paddingRight", "paddingBottom", "paddingLeft",
    "fontStyle", "fontVariant", "fontWeight", "fontStretch", "fontSize", "lineHeight", "fontFamily", "textAlign",
    "textTransform", "textIndent", "letterSpacing", "wordSpacing", "tabSize", "whiteSpace", "wordWrap", "wordBreak"];

  function caretRect(el) {
    if (isField(el)) {
      const pos = el.selectionStart, style = getComputedStyle(el);
      const div = document.createElement("div"), s = div.style;
      s.position = "absolute"; s.visibility = "hidden"; s.top = "0"; s.left = "-9999px";
      for (const p of MIRROR) s[p] = style[p];
      s.whiteSpace = el.tagName === "INPUT" ? "pre" : "pre-wrap";
      if (el.tagName === "TEXTAREA") s.wordWrap = "break-word";
      div.textContent = el.value.slice(0, pos);
      const mark = document.createElement("span");
      mark.textContent = el.value.slice(pos) || ".";
      div.appendChild(mark);
      document.documentElement.appendChild(div);
      const lineH = parseFloat(style.lineHeight) || parseFloat(style.fontSize) * 1.25;
      const top = mark.offsetTop + parseFloat(style.borderTopWidth);
      const left = mark.offsetLeft + parseFloat(style.borderLeftWidth);
      div.remove();
      const b = el.getBoundingClientRect();
      const x = Math.min(Math.max(b.left + left - el.scrollLeft, b.left), b.right);
      const y = b.top + top - el.scrollTop;
      return { left: x, top: y, bottom: y + lineH };
    }
    const sel = getSelection();
    const r = sel.getRangeAt(0).cloneRange();
    r.collapse(true);
    const rect = r.getClientRects()[0];
    if (rect) return { left: rect.left, top: rect.top, bottom: rect.bottom };
    let node = r.startContainer;                     // empty line: use the element the caret sits in
    if (node.nodeType === 1 && node.childNodes[r.startOffset]) node = node.childNodes[r.startOffset];
    const b = (node.nodeType === 1 ? node : node.parentElement || el).getBoundingClientRect();
    const lineH = parseFloat(getComputedStyle(el).lineHeight) || 18;
    return { left: b.left, top: b.top, bottom: b.top + lineH };
  }

  // Replace the `del` characters before the caret with `text`, the way typing would (keeps undo
  // and lets sites like Gmail or React apps notice the change).
  function replaceBefore(el, del, text) {
    if (isField(el)) {
      const pos = el.selectionStart;
      el.setSelectionRange(pos - del, pos);
      if (!document.execCommand("insertText", false, text)) {
        el.setRangeText(text, pos - del, pos, "end");
        el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: text }));
      }
      return;
    }
    const sel = getSelection();
    for (let i = 0; i < del; i++) sel.modify("extend", "backward", "character");
    document.execCommand("insertText", false, text);
  }

  // ---------------------------------------------------------------- the suggestion strip
  function buildStrip() {
    const host = document.createElement("div");
    host.style.cssText = "all:initial;position:fixed;z-index:2147483647;left:0;top:0;display:none;";
    const root = host.attachShadow({ mode: "closed" });
    root.innerHTML = `<style>
      .pill{display:flex;flex-direction:column;gap:1px;padding:5px;background:#2b2e35;color:#fff;
        border:1px solid #4b505a;border-radius:12px;box-shadow:0 6px 20px rgba(0,0,0,.28);min-width:120px;
        font:14px/1.3 "Segoe UI",system-ui,-apple-system,sans-serif;white-space:nowrap;user-select:none}
      .w{cursor:pointer;display:flex;gap:8px;align-items:baseline;padding:3px 8px;border-radius:7px}
      .w b{color:#8f96a3;font-size:11px;font-weight:600;width:10px;text-align:right}
      .w.sel{background:#41464f}
      .w:hover{background:#383c44}
      .note{color:#c9ced6;padding:2px 6px}
      .grip{cursor:move;color:#8f96a3;font-size:9px;text-align:center;letter-spacing:3px;line-height:8px;padding:0 0 3px}
    </style><div class="pill"></div>`;
    box = root.querySelector(".pill");
    box.addEventListener("mousedown", (e) => {        // keep focus in the text box
      e.preventDefault();
      if (e.target.closest(".grip")) { startDrag(e); return; }
      const w = e.target.closest(".w");
      if (w) accept(+w.dataset.i);
    });
    box.addEventListener("dblclick", (e) => {         // double-click the grip: back under the cursor
      if (!e.target.closest(".grip")) return;
      dragOffset = null;
      if (lastDefault) { strip.style.left = `${lastDefault.left}px`; strip.style.top = `${lastDefault.top}px`; }
    });
    document.documentElement.appendChild(host);
    return host;
  }

  function place(rect) {
    strip.style.display = "block";
    const h = strip.offsetHeight, w = strip.offsetWidth;
    let top = rect.bottom + 6;
    if (top + h > innerHeight - 4) top = rect.top - h - 6;
    let left = Math.max(4, Math.min(rect.left - 8, innerWidth - w - 4));
    top = Math.max(4, top);
    lastDefault = { left, top };
    if (dragOffset) {                                 // you moved it: keep it there, relative to the caret
      left = Math.max(4, Math.min(left + dragOffset.dx, innerWidth - w - 4));
      top = Math.max(4, Math.min(top + dragOffset.dy, innerHeight - h - 4));
    }
    strip.style.top = `${top}px`;
    strip.style.left = `${left}px`;
  }

  function startDrag(e) {
    dragging = { x: e.clientX, y: e.clientY, left: parseFloat(strip.style.left), top: parseFloat(strip.style.top) };
    const move = (ev) => {
      strip.style.left = `${dragging.left + ev.clientX - dragging.x}px`;
      strip.style.top = `${dragging.top + ev.clientY - dragging.y}px`;
    };
    const up = () => {
      document.removeEventListener("mousemove", move, true);
      document.removeEventListener("mouseup", up, true);
      if (lastDefault) dragOffset = { dx: parseFloat(strip.style.left) - lastDefault.left,
                                      dy: parseFloat(strip.style.top) - lastDefault.top };
      dragging = null;
    };
    document.addEventListener("mousemove", move, true);
    document.addEventListener("mouseup", up, true);
  }

  function show(words, rect, el) {
    if (!strip) strip = buildStrip();
    box.textContent = "";
    const grip = document.createElement("div");
    grip.className = "grip"; grip.textContent = "• • •"; grip.title = "Drag to move · double-click to reset";
    box.appendChild(grip);
    words.forEach(([word], i) => {
      const w = document.createElement("div");
      w.className = "w"; w.dataset.i = i;
      const n = document.createElement("b"); n.textContent = String(i + 1);
      const s = document.createElement("span"); s.textContent = word + (words[i][1].startsWith("\x08") ? "  ✎" : "");
      w.append(n, s); box.appendChild(w);
    });
    shown = words; sel = 0;
    paintSel();
    place(rect);
    ghostFor = { el, rect };
    drawGhost();
  }

  function paintSel() {
    [...box.querySelectorAll(".w")].forEach((w, i) => w.classList.toggle("sel", i === sel && shown.length > 1));
  }

  function move(step) {
    if (!shown.length) return;
    sel = (sel + step + shown.length) % shown.length;
    paintSel();
    drawGhost();
  }

  // Grey completion right after the caret, in the text box's own font. Only when nothing follows the
  // caret on that line (otherwise it would cover your text).
  function drawGhost() {
    if (!ghost) {
      ghost = document.createElement("div");
      ghost.style.cssText = "all:initial;position:fixed;z-index:2147483646;pointer-events:none;white-space:pre;display:none;";
      document.documentElement.appendChild(ghost);
    }
    const g = ghostFor, item = shown[sel];
    const after = g && g.el ? textAfter(g.el) : "x";
    if (!g || !item || item[1].startsWith("\x08") || (after && !/^\s*(\n|$)/.test(after))) { ghost.style.display = "none"; return; }
    const cs = getComputedStyle(g.el);
    ghost.textContent = item[1].replace(/ $/, "");
    ghost.style.font = cs.font;
    ghost.style.letterSpacing = cs.letterSpacing;
    ghost.style.color = "rgba(128,134,142,.85)";
    ghost.style.left = `${g.rect.left}px`;
    ghost.style.display = "block";
    const h = ghost.offsetHeight;
    ghost.style.top = `${g.rect.top + (g.rect.bottom - g.rect.top - h) / 2}px`;
  }

  function note(text, rect) {
    if (!strip) strip = buildStrip();
    box.innerHTML = "";
    const s = document.createElement("span"); s.className = "note"; s.textContent = text;
    box.appendChild(s);
    shown = []; ghostFor = null;
    if (ghost) ghost.style.display = "none";
    place(rect);
    const g = ++gen;
    setTimeout(() => { if (g === gen) hide(); }, 1200);
  }

  function hide() {
    shown = []; ghostFor = null;
    if (strip) strip.style.display = "none";
    if (ghost) ghost.style.display = "none";
  }

  // ---------------------------------------------------------------- predict
  function schedule(delay = DELAY) {
    hide();
    const g = ++gen;
    clearTimeout(timer);
    timer = setTimeout(() => predict(g), delay);
  }

  async function predict(g) {
    await refreshState();
    if (!active() || g !== gen) return;
    const el = target();
    if (!el) return;
    const text = textBefore(el);
    if (!text || !text.trim()) return;
    const { words } = await ask({ type: "predict", text, n: COUNT });
    if (g !== gen || !words || !words.length || target() !== el) return;
    show(words, caretRect(el), el);
  }

  function accept(i) {
    const el = target();
    const item = shown[i];
    if (!el || !item) return;
    const before = textBefore(el) || "";
    const fix = item[1].length - item[1].replace(/^\x08+/, "").length;   // a fix replaces the typed letters
    const text = item[1].slice(fix);
    hide();
    replaceBefore(el, fix, text);
    learnFrom(before.slice(0, before.length - fix) + text);
    schedule(0);                                     // suggest the word after it straight away
  }

  // ---------------------------------------------------------------- learning & auto-correct
  // Last finished word and the one before it; null for the first word in a box (could be a name or
  // username) or anything glued to symbols.
  function lastWord(text) {
    const m = text.match(/([\p{L}'’\-]+)$/u);
    if (!m) return null;
    const before = text.slice(0, text.length - m[1].length);
    if (!before.trim() || !/\s$/.test(before)) return null;
    const tail = before.trimEnd();
    const pm = tail.match(/([\p{L}'’\-]+)$/u);
    const start = ".!?\n".includes(tail.slice(-1));
    const prev = pm && !start ? pm[1].toLowerCase() : null;
    return { word: m[1], prev, start };
  }

  function learnFrom(textEndingInSpace) {
    const w = lastWord(textEndingInSpace.replace(/\s$/, ""));
    if (!w || /\d/.test(w.word)) return;
    const capital = w.word[0] !== w.word[0].toLowerCase();
    if (capital && !w.start && w.word !== "I" && !/^I['’]/.test(w.word)) return;   // a name or place
    ask({ type: "learn", prev: w.prev, word: w.word.toLowerCase().replace("’", "'") });
  }

  async function onSpace(e, el) {
    const before = textBefore(el);
    if (before == null) return;
    const w = lastWord(before);
    if (!w) return;
    if (!/^[a-z]+$/.test(w.word)) { learnFrom(before + " "); return; }   // only lowercase words get corrected
    e.preventDefault();                              // hold the space until we know
    pending = true;
    let fix = null;
    try { ({ fix } = await ask({ type: "correct", word: w.word, prev: w.prev })); } catch (err) {}
    pending = false;
    if (fix && fix !== w.word) {
      replaceBefore(el, w.word.length, fix + " ");
      lastFix = { el, word: w.word, fix, after: textBefore(el) };
      learnFrom((before.slice(0, before.length - w.word.length) + fix) + " ");
    } else {
      replaceBefore(el, 0, " ");
      learnFrom(before + " ");
    }
    if (queued) { replaceBefore(el, 0, queued); queued = ""; }
    schedule();
  }

  function undoFix(e, el) {
    const f = lastFix;
    lastFix = null;
    if (!f || f.el !== el || textBefore(el) !== f.after) return false;
    e.preventDefault();
    replaceBefore(el, f.fix.length + 1, f.word);
    ask({ type: "keep", word: f.word });
    hide();
    return true;
  }

  // ---------------------------------------------------------------- double-tap Ctrl
  let ctrlDown = 0, ctrlClean = false, lastTap = 0;
  function ctrlKey(e) {
    if (e.type === "keydown") {
      if (e.key === "Control") { if (!e.repeat) { ctrlDown = Date.now(); ctrlClean = true; } }
      else { ctrlClean = false; lastTap = 0; }
      return;
    }
    if (e.key !== "Control" || !ctrlDown) return;
    const now = Date.now(), tap = ctrlClean && now - ctrlDown <= 300;
    ctrlDown = 0;
    if (!tap) { lastTap = 0; return; }
    if (now - lastTap <= 400) { lastTap = 0; toggle(); } else lastTap = now;
  }

  async function toggle() {
    const r = await ask({ type: "toggle" });
    state.enabled = r.enabled !== false;
    state.checked = Date.now();
    const el = target();
    const rect = el ? caretRect(el) : { left: innerWidth / 2 - 80, top: 40, bottom: 60 };
    hide();
    note(`Word prediction ${state.enabled ? "on" : "off"}`, rect);
  }

  // ---------------------------------------------------------------- events
  document.addEventListener("keydown", (e) => {
    ctrlKey(e);
    if (e.isComposing || !active()) return;
    const el = target();
    if (!el) { hide(); return; }
    if (pending && e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault(); queued += e.key; return;   // keep keys in order while a correction is pending
    }
    const plain = !(e.ctrlKey || e.metaKey || e.altKey);
    if (shown.length && plain && !e.shiftKey) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault(); e.stopPropagation(); move(e.key === "ArrowDown" ? 1 : -1); return;
      }
      const m = /^(?:Digit|Numpad)([1-5])$/.exec(e.code);
      const i = m ? +m[1] - 1 : (e.key === "Tab" || e.key === "Enter") ? sel : -1;
      if (i >= 0 && i < shown.length) { e.preventDefault(); e.stopPropagation(); accept(i); return; }
    }
    if (e.key === "Escape") { hide(); return; }
    if (e.key === "Backspace" && plain && undoFix(e, el)) return;
    if (e.key !== "Backspace") lastFix = null;
    if (e.key === " " && plain) { onSpace(e, el); return; }
    if (e.key === "Enter" && plain) { const b = textBefore(el); if (b) learnFrom(b + "\n"); }
  }, true);

  document.addEventListener("keyup", ctrlKey, true);
  document.addEventListener("input", (e) => {         // after a space the next word is wanted at once
    if (active() && !e.isComposing && target()) schedule(e.data === " " ? 0 : DELAY);
  }, true);
  document.addEventListener("focusin", () => { hide(); refreshState(); }, true);
  document.addEventListener("mousedown", (e) => { if (strip && e.target !== strip && !dragging) hide(); lastFix = null; }, true);
  window.addEventListener("scroll", hide, true);
  window.addEventListener("blur", hide);
  refreshState(true);
})();
