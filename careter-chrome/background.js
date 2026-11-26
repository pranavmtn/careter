// Careter for Chrome: the engine lives here (one copy for all tabs). Pages ask it for suggestions,
// corrections, and tell it which words you finished so it can learn. Your learned words are kept in
// this browser profile only (chrome.storage.local); nothing is sent anywhere.
importScripts("engine.js");
const E = self.CareterEngine;
const NORVIG = "https://norvig.com/ngrams/";
const DESKTOP = "http://127.0.0.1:47822/ping";

let ready = null, ngram = null, speller = null, user = null;
let saveTimer = null, desktop = { running: false, checked: 0 };

async function loadModel() {
  const { model } = await chrome.storage.local.get("model");
  if (model && model.version === 1) return model;
  let built = null;
  try {                                            // a word file bundled with the extension, if any
    const r = await fetch(chrome.runtime.getURL("model.json"));
    if (r.ok) built = await r.json();
  } catch (e) {}
  if (!built) {                                    // first run: download the word lists once (~10 MB)
    const [a, b] = await Promise.all(["count_1w.txt", "count_2w.txt"].map((f) => fetch(NORVIG + f).then((r) => {
      if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`);
      return r.text();
    })));
    built = E.buildModel(a, b);
  }
  await chrome.storage.local.set({ model: built });
  return built;
}

async function load() {
  const model = await loadModel();
  const { user: saved } = await chrome.storage.local.get("user");
  ngram = new E.NGram(model);
  user = new E.UserModel(saved);
  speller = new E.Speller(ngram, user);
  if (user.purge((w) => speller.isTypo(w))) saveSoon();   // words saved earlier that are really typos
}

function ensure() {
  if (!ready) ready = load().catch((e) => { ready = null; throw e; });
  return ready;
}

function saveSoon() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => chrome.storage.local.set({ user: user.toJSON() }), 2000);
}

async function isEnabled() {
  const { enabled } = await chrome.storage.local.get("enabled");
  return enabled !== false;
}

async function desktopRunning() {
  // If the Careter desktop app is running, it already handles prediction everywhere (with the
  // helper extension), so this extension stays quiet to avoid two suggestion strips.
  if (Date.now() - desktop.checked < 15000) return desktop.running;
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), 400);
  try {
    const r = await fetch(DESKTOP, { signal: ctl.signal });
    desktop.running = r.ok;
  } catch (e) {
    desktop.running = false;
  }
  clearTimeout(t);
  desktop.checked = Date.now();
  return desktop.running;
}

async function setEnabled(on) {
  await chrome.storage.local.set({ enabled: on });
  chrome.action.setBadgeText({ text: on ? "" : "OFF" });
  chrome.action.setBadgeBackgroundColor({ color: "#6b7280" });
}

chrome.action.onClicked.addListener(async () => setEnabled(!(await isEnabled())));
chrome.runtime.onStartup.addListener(async () => setEnabled(await isEnabled()));
chrome.runtime.onInstalled.addListener(async () => { setEnabled(await isEnabled()); ensure().catch(() => {}); });

async function handle(msg) {
  switch (msg.type) {
    case "state":
      return { enabled: await isEnabled(), desktop: await desktopRunning() };
    case "toggle": {
      const on = !(await isEnabled());
      await setEnabled(on);
      return { enabled: on };
    }
    case "predict":
      if (!E.worthPredicting(msg.text)) return { words: [] };   // sentence start: wait for a letter
      await ensure();
      return { words: E.personalPredict(ngram, user, msg.text, msg.n || 5) };
    case "correct":
      await ensure();
      return { fix: speller.correct(msg.word, msg.prev) };
    case "learn":
      await ensure();
      if (speller.isTypo(msg.word, msg.prev || null)) return {};   // a typo: never learn a wrong word
      // only a real word gets the "typed instead of the suggestions" boost
      user.learn(msg.prev || null, msg.word, msg.weight === 2 && ngram.rank.has(msg.word) ? 2 : 1);
      saveSoon();
      return {};
    case "keep":                                   // you undid a correction: never correct this word
      await ensure();
      user.keep.add(msg.word);
      saveSoon();
      return {};
    case "forget":
      await ensure();
      user = new E.UserModel();
      speller.user = user;
      await chrome.storage.local.remove("user");
      return {};
  }
  return {};
}

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  if (!msg || !msg.careter) return;
  handle(msg).then(reply, (e) => reply({ error: String(e && e.message || e) }));
  return true;                                     // reply asynchronously
});
