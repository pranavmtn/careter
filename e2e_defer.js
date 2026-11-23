const puppeteer = require("/opt/npm-tools/node_modules/puppeteer");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  const ext = "/tmp/ext";
  const browser = await puppeteer.launch({ executablePath: "/usr/bin/google-chrome", headless: true,
    args: [`--disable-extensions-except=${ext}`, `--load-extension=${ext}`, "--no-sandbox"] });
  const page = await browser.newPage();
  await page.goto("http://127.0.0.1:8765/"); await sleep(1500);
  await page.click("#ta"); await page.keyboard.type("I will send you the ", { delay: 30 }); await sleep(700);
  const strip = await page.evaluate(() => { const h = [...document.documentElement.children].find((e) => e.shadowRoot); return h && h.style.display !== "none"; });
  await page.keyboard.type("recieve ", { delay: 30 }); await sleep(300);
  console.log("desktop running -> strip shown:", !!strip, "| autocorrect:", JSON.stringify(await page.$eval("#ta", (e) => e.value)));
  await browser.close();
})();
