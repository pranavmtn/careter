// Relays caret updates from pages to the Careter app on this PC. Nothing goes anywhere else.
const CARETER = "http://127.0.0.1:47822/caret";

chrome.runtime.onMessage.addListener((msg) => {
  if (!msg || msg.type !== "careter-caret") return;
  fetch(CARETER, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(msg.data),
  }).catch(() => {});            // Careter not running: ignore
});
