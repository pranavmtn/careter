// Careter caret helper: when you type in a text box, sends the ~300 characters before the caret and
// the caret's position on the page to the Careter app on this PC. Never sends passwords, page
// addresses, or anything after the caret.
(() => {
  const MAX = 300;
  let timer = null;

  function editableTarget() {
    let el = document.activeElement;
    while (el && el.shadowRoot && el.shadowRoot.activeElement) el = el.shadowRoot.activeElement;
    if (!el) return null;
    if (el.tagName === "TEXTAREA") return el;
    if (el.tagName === "INPUT") {
      const t = (el.type || "text").toLowerCase();
      return ["text", "search", "email", "url", "tel", "password", ""].includes(t) ? el : null;
    }
    return el.isContentEditable ? el : null;
  }

  // Caret position inside <input>/<textarea>: copy the box's text styling into a hidden mirror div,
  // put a marker where the caret is, and measure the marker.
  const MIRROR_PROPS = ["boxSizing", "width", "height", "overflowX", "overflowY", "borderTopWidth",
    "borderRightWidth", "borderBottomWidth", "borderLeftWidth", "paddingTop", "paddingRight",
    "paddingBottom", "paddingLeft", "fontStyle", "fontVariant", "fontWeight", "fontStretch",
    "fontSize", "lineHeight", "fontFamily", "textAlign", "textTransform", "textIndent",
    "letterSpacing", "wordSpacing", "tabSize", "whiteSpace", "wordWrap", "wordBreak"];

  function inputCaretRect(el, pos) {
    const style = getComputedStyle(el);
    const div = document.createElement("div");
    const s = div.style;
    s.position = "absolute"; s.visibility = "hidden"; s.top = "0"; s.left = "-9999px";
    for (const p of MIRROR_PROPS) s[p] = style[p];
    if (el.tagName === "INPUT") { s.whiteSpace = "pre"; s.height = style.height; }
    else { s.whiteSpace = "pre-wrap"; s.wordWrap = "break-word"; }
    div.textContent = el.value.slice(0, pos);
    const mark = document.createElement("span");
    mark.textContent = el.value.slice(pos) || ".";
    div.appendChild(mark);
    document.body.appendChild(div);
    const lineH = parseFloat(style.lineHeight) || parseFloat(style.fontSize) * 1.2;
    const top = mark.offsetTop + parseFloat(style.borderTopWidth);
    const left = mark.offsetLeft + parseFloat(style.borderLeftWidth);
    document.body.removeChild(div);
    const box = el.getBoundingClientRect();
    const x = box.left + left - el.scrollLeft;
    const y = box.top + top - el.scrollTop;
    return { left: Math.min(Math.max(x, box.left), box.right), top: y, bottom: y + lineH };
  }

  function editableCaret(el) {
    const sel = window.getSelection();
    if (!sel || !sel.rangeCount) return null;
    const range = sel.getRangeAt(0);
    if (!el.contains(range.startContainer)) return null;
    const before = document.createRange();
    before.selectNodeContents(el);
    before.setEnd(range.startContainer, range.startOffset);
    const text = before.toString().slice(-MAX);
    const caret = range.cloneRange();
    caret.collapse(true);
    let r = caret.getClientRects()[0];
    if (!r) {                                   // empty line: measure a temporary marker
      const mark = document.createElement("span");
      mark.textContent = "​";
      caret.insertNode(mark);
      r = mark.getBoundingClientRect();
      mark.remove();
      el.normalize();
    }
    return { text, rect: { left: r.left, top: r.top, bottom: r.bottom } };
  }

  // Offset of this frame inside the top page (same-origin frames only; otherwise unknown).
  function frameOffset() {
    let x = 0, y = 0, w = window;
    try {
      while (w !== w.top) {
        const fr = w.frameElement;
        if (!fr) return null;
        const b = fr.getBoundingClientRect();
        x += b.left + fr.clientLeft; y += b.top + fr.clientTop;
        w = w.parent;
      }
    } catch (e) { return null; }
    return { x, y };
  }

  function report() {
    const el = editableTarget();
    if (!el) return;
    const isPassword = el.tagName === "INPUT" && (el.type || "").toLowerCase() === "password";
    let data;
    if (isPassword) {
      data = { password: true };                // never the text
    } else if (el.tagName === "INPUT" || el.tagName === "TEXTAREA") {
      if (el.selectionStart == null) return;
      const pos = el.selectionStart;
      data = { text: el.value.slice(Math.max(0, pos - MAX), pos), rect: inputCaretRect(el, pos) };
    } else {
      data = editableCaret(el);
      if (!data) return;
    }
    if (data.rect) {
      const off = frameOffset();
      if (off) { data.rect.left += off.x; data.rect.top += off.y; data.rect.bottom += off.y; }
      else data.rect = null;                    // cross-origin frame: position unknown, text still useful
    }
    data.dpr = window.devicePixelRatio || 1;
    data.ts = Date.now();
    try { chrome.runtime.sendMessage({ type: "careter-caret", data }); } catch (e) {}
  }

  function schedule() {
    clearTimeout(timer);
    timer = setTimeout(report, 40);
  }

  document.addEventListener("input", schedule, true);
  document.addEventListener("keyup", schedule, true);
  document.addEventListener("click", schedule, true);
  document.addEventListener("focusin", schedule, true);
  document.addEventListener("selectionchange", schedule, true);
})();
