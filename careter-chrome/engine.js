// Careter engine for the browser: word prediction, auto-correct and learning.
// A JavaScript port of predict.py (NGramPredictor, Speller, UserModel). Pure functions + small classes,
// no browser APIs, so it also runs under Node for tests.
"use strict";

const MAX_CHARS = 400;
const WORD_CH = "(?:[\\p{L}\\p{N}]|['’\\-])";
const PARTIAL_RE = new RegExp(WORD_CH + "*$", "u");
const TAIL_WORD_RE = new RegExp("(" + WORD_CH + "+)$", "u");
const SENTENCE_END = ".!?\n";

// Personal names: never suggested (still treated as correctly spelled). Generated from names.py.
const NAMES = new Set(["aaron", "abhijith", "abhishek", "abigail", "abraham", "adam", "adams", "adarsh", "aditya", "adrian", "agarwal", "ahmed", "aiden", "ajay", "akhil", "akshay", "alan", "albert", "alex", "alexander", "alexis", "alice", "alicia", "allen", "allison", "alvarez", "amanda", "amber", "amit", "amy", "anand", "anderson", "andrea", "andrew", "angela", "anil", "anish", "anjali", "anjana", "ankit", "ann", "anna", "anne", "anoop", "anthony", "antonio", "anu", "aparna", "aravind", "arjun", "arun", "ashley", "ashwin", "aswathy", "aswin", "athira", "audrey", "austin", "ava", "ayesha", "banerjee", "barbara", "ben", "benjamin", "bennett", "betty", "beverly", "biju", "billy", "bose", "brandon", "brenda", "brian", "brittany", "brown", "bruce", "bryan", "caleb", "cameron", "carl", "carlos", "caroline", "castillo", "catherine", "charles", "charlotte", "chatterjee", "chavez", "cherian", "cheryl", "chloe", "chopra", "chris", "christina", "christine", "christopher", "cindy", "claire", "clark", "collins", "cruz", "cynthia", "dan", "daniel", "danielle", "dave", "david", "davies", "davis", "debbie", "deborah", "debra", "deepa", "deepak", "denise", "dennis", "deshmukh", "devika", "diana", "diane", "diaz", "divya", "donald", "donna", "doris", "dorothy", "douglas", "dylan", "edward", "edwards", "elijah", "elizabeth", "ella", "emily", "emma", "eric", "ethan", "eugene", "evans", "evelyn", "fathima", "flores", "gabriel", "garcia", "gary", "george", "gerald", "ghosh", "gloria", "gokul", "gomez", "gonzalez", "gopika", "gregory", "gupta", "gutierrez", "hannah", "hari", "harold", "harris", "heather", "helen", "henry", "hernandez", "howard", "hrishikesh", "hughes", "hussain", "isabella", "iyer", "jackson", "jacob", "jacobs", "jacqueline", "jain", "james", "janet", "janice", "jason", "jayan", "jayden", "jeffrey", "jennifer", "jenny", "jeremy", "jerry", "jesse", "jessica", "jessie", "jibin", "jim", "jimenez", "jinu", "jishnu", "jithin", "joan", "joe", "john", "johnny", "johnson", "jomon", "jonathan", "jones", "jordan", "jose", "joseph", "josh", "joshi", "joshua", "joyce", "juan", "judith", "judy", "julia", "julie", "justin", "kapoor", "karen", "kate", "katherine", "kathleen", "kathryn", "kathy", "kavya", "kayla", "keerthi", "keith", "kelly", "kenneth", "kevin", "kimberly", "kiran", "kulkarni", "kumar", "kurian", "kurup", "kyle", "larry", "laura", "lauren", "lekha", "lewis", "liam", "linda", "linu", "lisa", "logan", "lopez", "lori", "louis", "lucas", "madison", "mahesh", "malhotra", "manoj", "manu", "margaret", "maria", "marie", "marilyn", "martha", "martin", "martinez", "mary", "mathew", "matt", "matthew", "meera", "megan", "mehta", "melissa", "mendoza", "menon", "mia", "michael", "michel", "michelle", "midhun", "mike", "miller", "mishra", "mitchell", "mohammad", "mohammed", "moore", "morales", "morgan", "morris", "muhammed", "mukherjee", "murphy", "myers", "nair", "nandana", "nandu", "natalie", "nathan", "neha", "nelson", "nguyen", "nicholas", "nick", "nicole", "nikhil", "nimisha", "nithya", "noah", "oliver", "olivia", "ortiz", "pamela", "pandey", "parker", "parvathy", "patel", "patricia", "patrick", "paul", "perez", "peter", "peterson", "phil", "philip", "phillips", "pillai", "pooja", "pranav", "priya", "rachel", "rahul", "rajesh", "raju", "ralph", "ramesh", "ramos", "randy", "rao", "ravi", "raymond", "rebecca", "reddy", "rejith", "renjith", "reshma", "revathi", "reyes", "richard", "richardson", "robert", "roberts", "robinson", "rodriguez", "roger", "rogers", "rohit", "ronald", "ross", "ruiz", "russell", "ruth", "ryan", "sachin", "saji", "sajith", "samantha", "samuel", "sanchez", "sandra", "sanjay", "sara", "sarah", "scott", "sean", "shah", "shaji", "sharma", "sharon", "shibu", "shirley", "shyam", "singh", "sinu", "smith", "sneha", "sophia", "sophie", "sreeja", "sreejith", "sreekumar", "sruthi", "stephanie", "stephen", "steve", "steven", "stewart", "sujith", "sunil", "suresh", "susan", "taylor", "teresa", "theresa", "thomas", "thompson", "tim", "timothy", "tinu", "tiwari", "tom", "torres", "tyler", "unni", "varghese", "varma", "varun", "verma", "victoria", "vijay", "vincent", "vinod", "vipin", "vivek", "walter", "watson", "wayne", "william", "williams", "wilson", "wright", "yadav", "zachary", "zoe"]);

// Everyday work/tech words: ranked higher when completing a word, never auto-corrected. From vocab.py.
const BOOST = new Set(["access", "accuracy", "admin", "agenda", "agent", "agents", "ai", "alert", "alerts", "analytics", "android", "api", "apis", "app", "approval", "approve", "approved", "apps", "asap", "assignee", "attached", "attachment", "attachments", "autocomplete", "automation", "availability", "available", "aws", "azure", "backend", "backup", "blocked", "blocker", "branch", "branches", "browser", "budget", "bug", "bugs", "build", "builds", "cache", "caching", "calendar", "cancel", "cancelled", "cart", "chatbot", "checklist", "checkout", "chrome", "clarification", "clarify", "client", "clients", "cloud", "codebase", "colleague", "colleagues", "command", "commands", "commit", "commits", "component", "components", "config", "configuration", "confirm", "confirmation", "crash", "crashed", "cron", "csv", "customer", "customers", "dashboard", "data", "database", "databases", "dataset", "deadline", "deadlines", "debug", "debugging", "delete", "deleted", "deliverable", "deliverables", "dependencies", "dependency", "deploy", "deployed", "deployment", "design", "designer", "desktop", "discussed", "discussion", "django", "dns", "docker", "docs", "document", "documentation", "documents", "domain", "download", "downloaded", "downtime", "drive", "duplicate", "edit", "edited", "email", "emails", "embedding", "embeddings", "endpoint", "endpoints", "eod", "error", "errors", "estimate", "estimates", "evening", "excel", "exception", "extension", "feature", "features", "feedback", "figma", "firebase", "flask", "folder", "followup", "forwarding", "frontend", "fyi", "github", "gitlab", "gmail", "gpt", "gst", "holiday", "hosting", "hotfix", "incident", "inference", "install", "installation", "integration", "interview", "invitation", "invite", "invoice", "invoices", "ios", "issue", "issues", "java", "javascript", "jira", "json", "keyboard", "kindly", "kotlin", "kpi", "kubernetes", "laptop", "latency", "layout", "leave", "link", "links", "linux", "llm", "localhost", "logging", "login", "logout", "logs", "macos", "manager", "meeting", "meetings", "merge", "merged", "metrics", "migrate", "migration", "milestone", "mobile", "mockup", "model", "models", "module", "modules", "mongodb", "monitoring", "morning", "mysql", "netlify", "nginx", "node", "noted", "notification", "notifications", "office", "onboarding", "optimize", "order", "orders", "otp", "outage", "outlook", "password", "patch", "payment", "payments", "pdf", "performance", "permission", "permissions", "pipeline", "please", "plugin", "portal", "portals", "postgres", "postpone", "prediction", "predictions", "presentation", "priority", "production", "project", "projects", "prompt", "prompts", "proposal", "prototype", "pull", "python", "qa", "queries", "query", "queue", "quotation", "quote", "react", "redis", "refactor", "refresh", "regarding", "regards", "regression", "release", "releases", "reload", "reminder", "remote", "remove", "removed", "rename", "reopen", "reopened", "repo", "report", "reporter", "reports", "repository", "request", "requests", "requirement", "requirements", "reschedule", "reset", "resolved", "responsive", "restore", "resume", "retry", "review", "reviewed", "roadmap", "salary", "sanity", "save", "saved", "scenario", "scenarios", "schedule", "scheduled", "schema", "scope", "screen", "screenshot", "script", "scripts", "server", "servers", "shared", "signup", "slack", "slides", "smoke", "spreadsheet", "sprint", "sql", "ssl", "staging", "stakeholder", "stakeholders", "standup", "submit", "submitted", "suggestion", "suggestions", "supabase", "swift", "sync", "task", "tasks", "tax", "team", "teammate", "teams", "telegram", "template", "templates", "terminal", "testcase", "testcases", "tester", "testing", "thanks", "ticket", "tickets", "timeout", "today", "token", "tokens", "tomorrow", "training", "typescript", "uat", "ui", "update", "updates", "upgrade", "upload", "uploaded", "uptime", "urgent", "url", "username", "ux", "vector", "vendor", "vendors", "vercel", "verification", "verify", "version", "versions", "webhook", "webhooks", "webpage", "website", "weekend", "whatsapp", "windows", "wireframe", "workflow", "workflows", "xml", "yaml", "yesterday", "zoom"]);

const COMMON_TYPOS = {"teh": "the", "hte": "the", "thw": "the", "tje": "the", "adn": "and", "nad": "and", "anf": "and", "taht": "that", "tht": "that", "thta": "that", "waht": "what", "whta": "what", "wiht": "with", "wtih": "with", "jsut": "just", "juts": "just", "thier": "their", "theyre": "they're", "becuase": "because", "becasue": "because", "beacuse": "because", "recieve": "receive", "recieved": "received", "beleive": "believe", "wierd": "weird", "freind": "friend", "freinds": "friends", "untill": "until", "occured": "occurred", "seperate": "separate", "definately": "definitely", "goverment": "government", "tommorow": "tomorrow", "tommorrow": "tomorrow", "tomorow": "tomorrow", "realy": "really", "alot": "a lot", "dont": "don't", "doesnt": "doesn't", "didnt": "didn't", "cant": "can't", "wont": "won't", "isnt": "isn't", "wasnt": "wasn't", "im": "I'm", "ive": "I've", "youre": "you're", "knwo": "know", "konw": "know", "nwo": "now", "ot": "to", "fo": "of", "fro": "for", "ofr": "for", "yuo": "you", "yoru": "your", "abotu": "about", "aboutt": "about", "woudl": "would", "coudl": "could", "shoudl": "should", "whihc": "which", "wehn": "when", "becaus": "because", "pleae": "please", "plese": "please", "thnaks": "thanks", "thansk": "thanks", "msot": "most", "frist": "first", "thign": "thing", "thigns": "things", "agian": "again"};

// (partial word being typed, previous word or null, at sentence start) or null where a suggestion
// makes no sense (empty box, right after punctuation like "end.").
function readContext(text) {
  text = text.slice(-MAX_CHARS).replace(/\u00a0/g, " ");   // rich-text boxes type a trailing space as nbsp
  const partial = text.match(PARTIAL_RE)[0];
  const prefix = text.slice(0, text.length - partial.length);
  if (!prefix.trim() && !partial) return null;
  if (prefix && !(/\s/.test(prefix.slice(-1)) || "([{\"'“‘/".includes(prefix.slice(-1)))) return null;
  const tail = prefix.replace(/[ \t]+$/, "");
  const start = !tail || SENTENCE_END.includes(tail.slice(-1));
  const m = tail.match(TAIL_WORD_RE);
  const prev = m && !start ? m[1].toLowerCase() : null;
  return { partial, prev, start };
}

// False where suggestions would just be generic filler ("The", "A", "I"): an empty box, a new line,
// the start of a sentence or right after a comma, with no letter typed yet: wait for the first letter.
function worthPredicting(text) {
  const ctx = readContext(text);
  return !!(ctx && (ctx.partial || ctx.prev));        // a letter typed, or a word right before
}

// [shown word, text to insert] for lowercase word w, keeping the case of what was typed.
function shape(w, partial, start) {
  const rest = w.slice(partial.length);
  let shown;
  if (w === "i" || w.startsWith("i'")) shown = "I" + w.slice(1);
  else if (partial) shown = partial + (partial === partial.toUpperCase() && partial.length > 1 && partial !== partial.toLowerCase() ? rest.toUpperCase() : rest);
  else if (start) shown = w.charAt(0).toUpperCase() + w.slice(1);
  else shown = w;
  return [shown, shown.slice(partial.length) + " "];
}

// Compact model: words sorted most-common first, bigrams as word indexes.
class NGram {
  constructor(model) {
    const known = new Set(model.words);
    this.extra = new Set([...BOOST].filter((w) => !known.has(w)));   // jira, hotfix...: never "fixed to"
    this.words = model.words.concat([...this.extra].sort());
    this.bi = model.bi;                               // { prevWord: [index, ...] }
    this.rank = new Map(this.words.map((w, i) => [w, i]));
    this.pre = new Map();                             // prefix (3-8 letters) -> most common words with it
    for (const w of this.words.slice(0, 30000)) for (let n = 3; n <= Math.min(w.length, 8); n++) {
      const k = w.slice(0, n);
      if (!this.pre.has(k)) this.pre.set(k, []);
      const lst = this.pre.get(k);
      if (lst.length < 6) lst.push(w);
    }
    this.by1 = new Map(); this.by2 = new Map();
    for (const w of this.words) {
      if (!this.by1.has(w[0])) this.by1.set(w[0], []);
      this.by1.get(w[0]).push(w);
      if (w.length > 1) {
        const k = w.slice(0, 2);
        if (!this.by2.has(k)) this.by2.set(k, []);
        this.by2.get(k).push(w);
      }
    }
  }
  after(prev) { return (this.bi[prev] || []).map((i) => this.words[i]); }
  predict(text, n = 5) {
    const ctx = readContext(text);
    if (!ctx) return [];
    const p = ctx.partial.toLowerCase();
    const picks = [];
    const add = (w) => { if (w.length > p.length && w.startsWith(p) && !picks.includes(w) && !NAMES.has(w)) picks.push(w); };
    if (ctx.prev) for (const w of this.after(ctx.prev)) { add(w); if (picks.length >= n) break; }
    if (picks.length < n) {
      const pool = p.length >= 2 ? this.by2.get(p.slice(0, 2)) || [] : p ? this.by1.get(p) || [] : this.words;
      if (p) {                                      // completing a word: work words rank higher (max 2)
        const cands = [];
        for (const w of pool) if (w.length > p.length && w.startsWith(p)) { cands.push(w); if (cands.length >= 300) break; }
        for (const w of BOOST) if (w.length > p.length && w.startsWith(p) && !cands.includes(w)) cands.push(w);
        const eff = (w) => (this.rank.has(w) ? this.rank.get(w) : 1e6) * (BOOST.has(w) ? 0.03 : 1);
        const ordered = cands.sort((a, b) => eff(a) - eff(b));
        let boosted = 0;
        for (const w of ordered) {
          if (BOOST.has(w)) { if (boosted >= 2) continue; boosted++; }
          add(w); if (picks.length >= n) break;
        }
        for (const w of ordered) { if (picks.length >= n) break; add(w); }
      } else {
        for (const w of pool) { add(w); if (picks.length >= n) break; }
      }
    }
    const out = picks.slice(0, n).map((w) => shape(w, ctx.partial, ctx.start));
    return this.withFixes(out, ctx.partial, ctx.start, n);
  }
  // Words you probably meant while typing a misspelled one ("craet" -> created, create).
  fixes(partial, limit = 3) {
    const p = partial.toLowerCase();
    if (p.length < 3 || !/^[a-z]+$/.test(p) || BOOST.has(p) || (this.rank.has(p) && this.rank.get(p) < 20000)) return [];
    const found = new Map();
    for (const e of edits1(p)) {
      if (e.length < 3 || e === p || e === p.slice(0, -1) || (e[0] !== p[0] && e.slice(0, 2) !== p[1] + p[0])) continue;
      for (const w of this.pre.get(e.slice(0, 8)) || [])
        if (w.startsWith(e) && !NAMES.has(w) && !w.startsWith(p)) {
          const r = this.rank.has(w) ? this.rank.get(w) : 1e9;
          if (!found.has(w) || found.get(w) > r) found.set(w, r);
        }
    }
    return [...found.keys()].sort((a, b) => found.get(a) - found.get(b)).slice(0, limit);
  }
  // Fixes go first when far more common than anything starting with what was typed, else last.
  // A fix replaces what was typed, so its text starts with a backspace (\x08) per typed letter.
  withFixes(out, partial, start, n) {
    const rk = (w) => (this.rank.has(w) ? this.rank.get(w) : 1e9);
    const bestTyped = Math.min(1e9, ...out.map((x) => rk(x[0].toLowerCase())));
    if (bestTyped < 5000) return out;
    const fixes = this.fixes(partial);
    if (!fixes.length) return out;
    const items = fixes.map((w) => {
      const shown = partial[0] && partial[0] !== partial[0].toLowerCase() ? w[0].toUpperCase() + w.slice(1) : w;
      return [shown, "\x08".repeat(partial.length) + shown + " "];
    });
    const merged = rk(fixes[0]) * 4 < bestTyped ? items.concat(out)
      : out.slice(0, out.length ? Math.max(n - items.length, n - 1) : 0).concat(items);
    const seen = new Set(), result = [];
    for (const it of merged) if (!seen.has(it[0].toLowerCase())) { seen.add(it[0].toLowerCase()); result.push(it); }
    return result.slice(0, n);
  }
}

// Manglish words & phrases (same list as the desktop app's vocab.py). Words are offered as you type
// and never auto-corrected; phrases suggest the next word ("pinne " -> kaanam).
const MANGLISH = ["sughamano", "sugham", "enthokke undu", "enthaa", "evide", "eppo", "aara", "enthina", "engane", "sheri", "aa", "aanu", "alla", "illa", "undu", "nanni", "pinne kaanam", "poyitt varaam", "machane", "machaa", "aliya", "chetta", "chechi", "mone", "mole", "pwoli", "kidu", "polichu", "adipoli", "scene illa", "scene aanu", "vibe aanu", "thallu", "thalli marikkal", "chali", "mass", "kalippu", "kalli", "pani kitti", "seen aakki", "ayyo", "ente ponno", "mathi", "vende", "venam", "pettennu", "pathukke", "njan varaam", "njan busy aanu", "call cheyyam", "message ayakk", "ready aayo", "ithu set aanu", "tension adikkanda", "free aano", "food kazhicho", "late aakum", "pinne paranja mathi", "onnum manasilayilla", "enikk ariyilla", "kuzhappam illa", "vere level", "download cheyyu", "share cheyyam", "delete cheythu", "kollaam", "ponnu", "chunk", "muthe", "kutta", "eda", "edi", "aashaan", "uncle", "aunty", "kashtam", "bore aanu", "ozhivu", "pokkam", "vaa", "po", "nokk", "para", "kelk", "irikk", "kazhikk", "kudikk", "urangu", "ezhunnelkk", "sathyam", "vishwasikk", "ishtam", "pedi", "sankadam", "santhosham", "poli sanam", "thakarthu", "kalakki", "pottan", "mandan", "kozhi", "kaasu", "panam", "vandi", "veedu", "chaya", "choru", "vellam", "mazha", "choodu", "thanupp"];
const MANG_ORDER = [], MANG_PAIRS = {};
for (const p of MANGLISH) {
  const ws = p.match(/[a-z']+/g) || [];
  for (const w of ws) if (!MANG_ORDER.includes(w)) MANG_ORDER.push(w);
  for (let i = 0; i + 1 < ws.length; i++) {
    const n = (MANG_PAIRS[ws[i]] = MANG_PAIRS[ws[i]] || []);
    if (!n.includes(ws[i + 1])) n.push(ws[i + 1]);
  }
}
const MANG_WORDS = new Set(MANG_ORDER);

// Words and word pairs YOU type. Plain object so it can be saved in chrome.storage.
class UserModel {
  constructor(data) {
    this.uni = (data && data.words) || {};
    this.bi = (data && data.pairs) || {};
    this.keep = new Set((data && data.keep) || []);   // words you undid: never corrected
    for (const w of Object.keys(this.uni)) if (NAMES.has(w)) delete this.uni[w];
    for (const a of Object.keys(this.bi)) {
      if (NAMES.has(a)) { delete this.bi[a]; continue; }
      for (const b of Object.keys(this.bi[a])) if (NAMES.has(b)) delete this.bi[a][b];
    }
  }
  // weight 2 = a word you typed instead of the suggestions shown: suggested first the very next time
  learn(prev, word, weight = 1) {
    if (NAMES.has(word)) return;
    if (NAMES.has(prev)) prev = null;
    this.uni[word] = (this.uni[word] || 0) + weight;
    if (prev) { const d = (this.bi[prev] = this.bi[prev] || {}); d[word] = (d[word] || 0) + weight; }
  }
  // forget learned words that are really typos (isTypo(word) -> true)
  purge(isTypo) {
    const bad = new Set();
    for (const w of Object.keys(this.uni)) if (isTypo(w)) bad.add(w);
    for (const a of Object.keys(this.bi)) {
      if (isTypo(a)) bad.add(a);
      for (const b of Object.keys(this.bi[a])) if (isTypo(b)) bad.add(b);
    }
    for (const w of bad) delete this.uni[w];
    for (const a of Object.keys(this.bi)) {
      if (bad.has(a)) { delete this.bi[a]; continue; }
      for (const b of Object.keys(this.bi[a])) if (bad.has(b)) delete this.bi[a][b];
    }
    return bad.size;
  }
  suggest(prev, p, n) {
    const out = [];
    const after = prev ? this.bi[prev] || {} : {};
    for (const [w, c] of Object.entries(after).sort((a, b) => b[1] - a[1]))
      if (c >= 2 && w.length > p.length && w.startsWith(p) && !out.includes(w) && !NAMES.has(w)) out.push(w);
    if (p) for (const [w, c] of Object.entries(this.uni).sort((a, b) => b[1] - a[1])) {
      if (out.length >= n) break;
      if (c >= 2 && w.length > p.length && w.startsWith(p) && !out.includes(w) && !NAMES.has(w)) out.push(w);
    }
    return out.slice(0, n);
  }
  manglish(prev, p, n = 2) {
    const out = (MANG_PAIRS[prev] || []).filter((w) => w.length > p.length && w.startsWith(p));
    if (p.length >= 2) {
      const more = MANG_ORDER.filter((w) => w.length > p.length && w.startsWith(p) && !out.includes(w));
      more.sort((a, b) => (this.uni[b] || 0) - (this.uni[a] || 0));
      out.push(...more);
    }
    return out.filter((w) => !NAMES.has(w)).slice(0, n);
  }
  count(w) { return this.uni[w] || 0; }
  toJSON() {
    let pairs = this.bi;
    let total = 0; for (const d of Object.values(pairs)) total += Object.keys(d).length;
    if (total > 60000) {                               // forget one-off pairs first
      pairs = {};
      for (const [a, d] of Object.entries(this.bi)) {
        const kept = Object.fromEntries(Object.entries(d).filter(([, c]) => c > 1));
        if (Object.keys(kept).length) pairs[a] = kept;
      }
      this.bi = pairs;
    }
    return { words: this.uni, pairs, keep: [...this.keep] };
  }
}

// Your most used words first (they may fill the list), then the general suggestions.
function personalPredict(ngram, user, text, n = 5, mine = 5) {     // your most used words first
  const base = ngram.predict(text, n);
  const ctx = readContext(text);
  if (!ctx || !user) return base;
  const out = [], seen = new Set();
  for (const w of user.suggest(ctx.prev, ctx.partial.toLowerCase(), Math.min(mine, n))) {
    const item = shape(w, ctx.partial, ctx.start);
    out.push(item); seen.add(item[0].toLowerCase());
  }
  // Manglish: right after a Manglish word its phrase comes first; otherwise one takes the last slot
  let mang = user.manglish(ctx.prev, ctx.partial.toLowerCase(), 2).filter((w) => !seen.has(w));
  if (mang.length && MANG_WORDS.has(ctx.prev)) {
    for (const w of mang) { const item = shape(w, ctx.partial, ctx.start); out.push(item); seen.add(item[0].toLowerCase()); }
    mang = [];
  }
  for (const item of base) if (!seen.has(item[0].toLowerCase()) && out.length < n) { out.push(item); seen.add(item[0].toLowerCase()); }
  if (mang.length && n > 1) return [...out.slice(0, n - 1), shape(mang[0], ctx.partial, ctx.start)];
  return out.slice(0, n);
}

// ---------------------------------------------------------------- spelling
const LETTERS = "abcdefghijklmnopqrstuvwxyz";
function edits1(w) {
  const out = new Set();
  for (let i = 0; i <= w.length; i++) {
    const a = w.slice(0, i), b = w.slice(i);
    if (b) out.add(a + b.slice(1));
    if (b.length > 1) out.add(a + b[1] + b[0] + b.slice(2));
    for (const c of LETTERS) { if (b) out.add(a + c + b.slice(1)); out.add(a + c + b); }
  }
  return out;
}
function deletes1(w) { const s = new Set([w]); for (let j = 0; j < w.length; j++) s.add(w.slice(0, j) + w.slice(j + 1)); return s; }
function osa(a, b) {
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)]);
  for (let j = 0; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) {
    const cost = a[i - 1] === b[j - 1] ? 0 : 1;
    d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
    if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1);
  }
  return d[a.length][b.length];
}
function letterPairs(w) { const s = new Set(); for (let i = 0; i < w.length - 1; i++) s.add(w.slice(i, i + 2)); return s; }

class Speller {
  constructor(ngram, user) {
    this.ng = ngram; this.user = user;
    this.COMMON = Math.min(40000, Math.floor(ngram.words.length * 0.8));
    this.RARE_TO = 3000; this.TARGET1 = 30000; this.EDIT2_WORDS = 15000; this.EDIT2_MIN_LEN = 7;
    this.LONG_MIN_LEN = 9; this.LONG_WORDS = 60000;
    this.dix = new Map();
    ngram.words.slice(0, this.EDIT2_WORDS).forEach((w, i) => {
      if (w.length >= this.EDIT2_MIN_LEN - 2) for (const k of deletes1(w)) { if (!this.dix.has(k)) this.dix.set(k, []); this.dix.get(k).push(i); }
    });
    this.long = new Map();
    for (const w of ngram.words.slice(0, this.LONG_WORDS)) if (w.length >= this.LONG_MIN_LEN - 3) {
      if (!this.long.has(w[0])) this.long.set(w[0], []);
      this.long.get(w[0]).push([w, letterPairs(w)]);
    }
  }
  // true if `word` is probably a typo, so it must never be learned (see predict.py Speller.is_typo)
  isTypo(word, prev) {
    if (this.correct(word, prev || null, false)) return true;
    if (word.length < 9 || !/^[a-z]+$/.test(word) || BOOST.has(word) || this.ng.rank.has(word)
        || (this.user && this.user.keep.has(word)) || MANG_WORDS.has(word)) return false;
    const mine = letterPairs(word), lim = Math.max(3, Math.round(word.length * 0.35));
    for (const [w, wp] of this.long.get(word[0]) || []) {
      if (Math.abs(w.length - word.length) > lim) continue;
      let inter = 0; for (const x of mine) if (wp.has(x)) inter++;
      if (inter / (mine.size + wp.size - inter) < 0.25) continue;
      if (osa(word, w) <= lim) return true;
    }
    return false;
  }
  correct(word, prev, trustLearned = true) {
    if (!/^[a-z]+$/.test(word)) return null;          // lowercase letters only: no names, ACRONYMS, numbers
    if (BOOST.has(word) || MANG_WORDS.has(word)) return null;   // an everyday work word, or Manglish
    if (this.user && (this.user.keep.has(word) || (trustLearned && this.user.count(word) >= 3))) return null;
    if (COMMON_TYPOS[word]) return COMMON_TYPOS[word];
    if (word.length < 3) return null;
    const r = this.ng.rank.get(word);
    if (r !== undefined && r < this.COMMON) return null;
    const first = (c) => c[0] === word[0] || c.slice(0, 2) === word[1] + word[0];
    const ok = (c, top) => { const rc = this.ng.rank.get(c); return rc !== undefined && rc < (r !== undefined ? this.RARE_TO : top) && first(c) && !this.ng.extra.has(c); };
    const follows = new Set(prev ? this.ng.after(prev) : []);
    const best = (cs) => cs.sort((a, b) => (follows.has(a) ? 0 : 1) - (follows.has(b) ? 0 : 1) || this.ng.rank.get(a) - this.ng.rank.get(b))[0];
    const e1 = edits1(word);
    let cands = [...e1].filter((c) => ok(c, this.TARGET1));
    if (cands.length) return best(cands);
    if (word.length < this.EDIT2_MIN_LEN) return null;
    const found = new Set();
    for (const e of e1) for (const k of deletes1(e)) for (const i of this.dix.get(k) || []) found.add(this.ng.words[i]);
    cands = [...found].filter((c) => ok(c, this.EDIT2_WORDS) && Math.abs(c.length - word.length) <= 2 && osa(word, c) <= 2);
    if (cands.length) return best(cands);
    if (word.length >= this.LONG_MIN_LEN) {
      const mine = letterPairs(word); let bestD = 4, list = [];
      for (const [w, wp] of this.long.get(word[0]) || []) {
        if (Math.abs(w.length - word.length) > 3 || !ok(w, this.LONG_WORDS)) continue;
        let inter = 0; for (const x of mine) if (wp.has(x)) inter++;
        if (inter / (mine.size + wp.size - inter) < 0.4) continue;
        const d = osa(word, w);
        if (d < bestD) { bestD = d; list = [w]; } else if (d === bestD) list.push(w);
      }
      if (bestD <= 3 && list.length) return best(list);
    }
    return null;
  }
}

// Build the compact model from Peter Norvig's count_1w.txt / count_2w.txt text.
function buildModel(text1w, text2w, maxWords = 50000, maxPrev = 20000, perPrev = 12) {
  const ok = (w) => /^[a-z]+$/.test(w) && (w.length > 1 || w === "a" || w === "i");
  const words = [];
  for (const line of text1w.split("\n")) {
    const w = line.split("\t")[0].toLowerCase();
    if (ok(w)) { words.push(w); if (words.length >= maxWords) break; }
  }
  const rank = new Map(words.map((w, i) => [w, i]));
  const tmp = {};
  for (const line of text2w.split("\n")) {
    const [pair, count] = line.split("\t");
    if (!count) continue;
    const [a, b] = pair.toLowerCase().split(" ");
    const ra = rank.get(a), rb = rank.get(b);
    if (ra === undefined || rb === undefined || ra >= maxPrev) continue;
    (tmp[a] = tmp[a] || []).push([+count, rb]);
  }
  const bi = {};
  for (const [a, list] of Object.entries(tmp)) bi[a] = list.sort((x, y) => y[0] - x[0]).slice(0, perPrev).map((x) => x[1]);
  return { version: 1, words, bi };
}

const CareterEngine = { NAMES, readContext, worthPredicting, shape, NGram, UserModel, Speller, personalPredict, buildModel, osa, COMMON_TYPOS };
if (typeof module !== "undefined") module.exports = CareterEngine;
else self.CareterEngine = CareterEngine;
