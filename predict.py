"""
Next-word prediction with a small local language model. No API key, runs on the CPU.

An LLM only ever predicts the next *token*. This module turns that into word suggestions:
  - "I will send you the ▮"  ->  next words: "details", "link", "file"
  - "I will send you the de▮" ->  completions of "de": "details", "document", "deck"
It reads the model's top next tokens, keeps the ones that start a word (and match what
you've typed of it), then extends each greedily until the word ends.
"""
import os
import re
import threading

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # harmless on Windows without symlinks

WORD = r"[^\W_]|['’\-]"                   # letters/digits plus ' ’ - inside words
PARTIAL_RE = re.compile(rf"(?:{WORD})*\Z")
LEAD_RE = re.compile(rf"(?:{WORD})+")
MAX_CHARS = 400                          # text before the caret fed to the model


class LocalLM:
    """A Hugging Face causal LM (use a *base* model, not a chat model)."""

    def __init__(self, name="HuggingFaceTB/SmolLM2-135M", threads=2, max_ctx_tokens=48):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
        except TypeError:                    # transformers older than 4.56
            self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
        self.model.eval()
        self.max_ctx = max_ctx_tokens
        bos = self.tok.bos_token_id
        self.bos = [bos] if bos is not None else []
        self.vocab = self._vocab_strings()

    def _vocab_strings(self):
        """Text of every token on its own, with its leading space kept (None = skip this token)."""
        n = len(self.tok)
        special = set(self.tok.all_special_ids)
        raw = self.tok.convert_ids_to_tokens(list(range(n)))
        sentencepiece = sum(1 for t in raw[:5000] if t and t.startswith("▁")) > 500
        if sentencepiece:                    # Llama-2/Gemma style: "▁the" means " the"
            texts = [None if (t is None or re.fullmatch(r"<0x[0-9A-F]{2}>", t)) else t.replace("▁", " ")
                     for t in raw]
        else:                                # byte-level BPE (GPT-2, SmolLM, Qwen, Llama-3)
            texts = self.tok.batch_decode([[i] for i in range(n)])
        return [None if (i in special or not t or "�" in t) else t for i, t in enumerate(texts)]

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def next_top(self, ids, k):
        return self.next_top_batch([ids], k)[0]

    def next_top_batch(self, rows, k):
        """For each row of token ids (all the same length): [(token_id, logprob)] top-k next tokens."""
        rows = [(self.bos + r)[-self.max_ctx:] or self.encode("\n") for r in rows]
        if len({len(r) for r in rows}) > 1:
            return [self.next_top_batch([r], k)[0] for r in rows]
        torch = self.torch
        with torch.inference_mode():
            logits = self.model(torch.tensor(rows)).logits[:, -1].float()
            v, i = torch.topk(torch.log_softmax(logits, -1), k)
        return [list(zip(ii, vv)) for ii, vv in zip(i.tolist(), v.tolist())]


class Predictor:
    """Turns next-token predictions into word suggestions. All candidate words grow one token per
    step, so each step is ONE batched forward pass however many candidates there are."""

    def __init__(self, lm, seeds=10, max_word_tokens=4):
        self.lm, self.seeds, self.max_word_tokens = lm, seeds, max_word_tokens
        self.lock = threading.Lock()

    def _top_batch(self, rows, k):
        if hasattr(self.lm, "next_top_batch"):
            return self.lm.next_top_batch(rows, k)
        return [self.lm.next_top(r, k) for r in rows]

    def predict(self, text, n=3):
        """Up to n suggestions for the caret at the end of text: [(word, text_to_insert)]."""
        text = text[-MAX_CHARS:]
        partial = PARTIAL_RE.search(text).group(0)
        prefix = text[: len(text) - len(partial)]
        if not prefix.strip() and not partial:
            return []                                   # empty box: nothing to go on
        if prefix.endswith(" "):
            prefix, lead = prefix[:-1], " "             # BPE keeps the space on the next word: " the"
        elif prefix == "" or prefix[-1] in "\n\t([{\"'“‘/":
            lead = ""
        else:
            return []                                   # right after punctuation like "end." : wait for a space
        ids = self.lm.encode(prefix) if prefix else []
        p = partial.lower()

        with self.lock:
            finished, active = {}, []
            for tok_ids, word, done, score in self._seeds(ids, lead, p):
                if done:
                    self._finish(finished, word, score, p)
                else:
                    active.append((tok_ids, word, score))
            for _ in range(self.max_word_tokens):
                if not active or self._settled(finished, active, n):
                    break
                tops = self._top_batch([ids + a[0] for a in active], 20)
                grown = []
                for (tok_ids, word, score), top in zip(active, tops):
                    step = self._step(word, p, top)
                    if step is None:
                        continue                        # can't match what was typed: drop
                    if step == "end":
                        self._finish(finished, word, score, p)
                        continue
                    tid, piece, lp, ends = step
                    cand = (tok_ids + [tid], word + piece, score + lp)
                    if ends:
                        self._finish(finished, cand[1], cand[2], p)
                    else:
                        grown.append(cand)
                active = grown
        ranked = sorted(finished.values(), key=lambda ws: -ws[1])[:n]
        return [(partial + w[len(partial):], w[len(partial):] + " ") for w, _ in ranked]

    @staticmethod
    def _consistent(word, p):
        w = word.lower()
        return w.startswith(p) or p.startswith(w)

    @staticmethod
    def _finish(finished, word, score, p):
        if len(word) > len(p) and word.lower().startswith(p):
            key = word.lower()
            if key not in finished or finished[key][1] < score:
                finished[key] = (word, score)

    @staticmethod
    def _settled(finished, active, n):
        """Scores only go down as words grow: stop once n finished words beat every open one."""
        if len(finished) < n:
            return False
        nth = sorted((s for _, s in finished.values()), reverse=True)[n - 1]
        return all(a[2] <= nth for a in active)

    def _step(self, word, p, top):
        """Best next piece for this word: (id, piece, logprob, word_ends), "end", or None."""
        for tid, lp in top:
            t = self.lm.vocab[tid] if tid < len(self.lm.vocab) else None
            if not t:
                continue
            m = LEAD_RE.match(t)
            if not m:                                   # space or punctuation: the word would end
                if len(word) > len(p):
                    return "end"
                continue                                # not yet: it must be longer than what's typed
            piece = m.group(0)
            if self._consistent(word + piece, p):
                return tid, piece, lp, len(piece) < len(t)
        return None

    def _seeds(self, ids, lead, p):
        """Top next tokens that start a word consistent with what's typed: (ids, word, done, logprob)."""
        out = []
        for tid, lp in self._top_batch([ids], 300)[0]:
            t = self.lm.vocab[tid] if tid < len(self.lm.vocab) else None
            if not t:
                continue
            if lead:
                if not t.startswith(lead):
                    continue
                body = t[len(lead):]
            else:
                if t[0].isspace():
                    continue
                body = t
            m = LEAD_RE.match(body)
            if not m or m.group(0)[0] in "'’-":
                continue
            word = m.group(0)
            done = len(word) < len(body)                # token runs past the end of the word
            if not self._consistent(word, p) or (done and len(word) <= len(p)):
                continue
            out.append(([tid], word, done, lp))
            if len(out) >= self.seeds:
                break
        return out


# ---------------------------------------------------------------- separate process
# The model runs in its own low-priority process: a busy model can never slow down your keyboard.

def _worker(conn, model_name):
    import sys
    import traceback
    for stream in ("stdout", "stderr"):                # pythonw has no console
        if getattr(sys, stream) is None:
            setattr(sys, stream, open(os.devnull, "w"))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)   # BELOW_NORMAL_PRIORITY_CLASS
    try:
        pred = Predictor(LocalLM(model_name))
        conn.send(("ready", None))
    except Exception:
        conn.send(("error", traceback.format_exc()))
        return
    while True:
        try:
            rid, text, n = conn.recv()
        except (EOFError, OSError):
            return
        try:
            words = pred.predict(text, n)
        except Exception:
            words = []
        conn.send(("words", (rid, words)))


class PredictorProcess:
    """Same predict() as Predictor, but the model lives in a child process."""

    def __init__(self, model_name):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(child, model_name), daemon=True)
        self.proc.start()
        kind, info = self.conn.recv()                  # waits for the model to load (or download)
        if kind == "error":
            raise RuntimeError(info)
        self.lock = threading.Lock()
        self.rid = 0

    def predict(self, text, n=3, cancelled=None, timeout=5.0):
        with self.lock:
            if cancelled and cancelled():
                return []                               # you typed more while we waited
            self.rid += 1
            rid = self.rid
            self.conn.send((rid, text, n))
            while self.conn.poll(timeout):
                kind, (got, words) = self.conn.recv()
                if got == rid:
                    return words                        # older, late answers are skipped
            return []


# ---------------------------------------------------------------- light engine: word counts (n-grams)
# No neural network: counts of common English words and word pairs (Peter Norvig's lists from the
# Google Web 1T data). Downloaded once (~10 MB), then answers in about a millisecond.

NGRAM_URL = "https://norvig.com/ngrams/"
SENTENCE_END = ".!?\n"


class NGramPredictor:
    def __init__(self, folder, max_words=100_000, per_prev=40, log=print):
        import pickle
        os.makedirs(folder, exist_ok=True)
        cache = os.path.join(folder, "ngrams.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                self.words, self.bi = pickle.load(f)
        else:
            files = {}
            for name in ("count_1w.txt", "count_2w.txt"):
                path = files[name] = os.path.join(folder, name)
                if not os.path.exists(path):
                    import urllib.request
                    log(f"predict: downloading {name}…")
                    urllib.request.urlretrieve(NGRAM_URL + name, path + ".part")
                    os.replace(path + ".part", path)
            self.words, self.bi = self._build(files, max_words, per_prev)
            with open(cache, "wb") as f:
                pickle.dump((self.words, self.bi), f, protocol=pickle.HIGHEST_PROTOCOL)
        self.lock = threading.Lock()
        self.by1, self.by2 = {}, {}                     # first letter(s) -> words, most common first
        for w in self.words:
            self.by1.setdefault(w[0], []).append(w)
            if len(w) > 1:
                self.by2.setdefault(w[:2], []).append(w)

    @staticmethod
    def _build(files, max_words, per_prev):
        def ok(w):
            return w.isalpha() and (len(w) > 1 or w in ("a", "i"))
        words = []
        with open(files["count_1w.txt"], encoding="utf-8", errors="ignore") as f:
            for line in f:
                w = line.split("\t", 1)[0].lower()
                if ok(w):
                    words.append(w)
                    if len(words) >= max_words:
                        break
        known = set(words)
        bi = {}
        with open(files["count_2w.txt"], encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    pair, count = line.rstrip("\n").split("\t")
                    a, b = pair.lower().split(" ")
                    count = int(count)
                except ValueError:
                    continue
                if a in known and b in known:
                    bi.setdefault(a, []).append((count, b))
        bi = {a: [b for _, b in sorted(v, reverse=True)[:per_prev]] for a, v in bi.items()}
        return words, bi

    def predict(self, text, n=3, **_):
        text = text[-MAX_CHARS:]
        partial = PARTIAL_RE.search(text).group(0)
        prefix = text[: len(text) - len(partial)]
        if not prefix.strip() and not partial:
            return []
        if prefix and not (prefix[-1].isspace() or prefix[-1] in "([{\"'“‘/"):
            return []                                   # glued to punctuation: "end." -> wait
        tail = prefix.rstrip(" \t")
        start = not tail or tail[-1] in SENTENCE_END
        m = re.search(rf"((?:{WORD})+)\Z", tail)
        prev = m.group(1).lower() if (m and not start) else None
        p = partial.lower()

        picks = []
        def add(w):
            if len(w) > len(p) and w.startswith(p) and w not in picks:
                picks.append(w)
        for w in self.bi.get(prev, []) if prev else []:
            add(w)
            if len(picks) >= n:
                break
        if len(picks) < n:
            pool = self.by2.get(p[:2], []) if len(p) >= 2 else self.by1.get(p, []) if p else self.words
            for w in pool:
                add(w)
                if len(picks) >= n:
                    break

        out = []
        for w in picks[:n]:
            rest = w[len(p):]
            if w == "i" or w.startswith("i'"):
                shown = "I" + w[1:]
            elif partial:
                shown = partial + (rest.upper() if partial.isupper() and len(partial) > 1 else rest)
            elif start:
                shown = w[:1].upper() + w[1:]
            else:
                shown = w
            out.append((shown, shown[len(partial):] + " "))
        return out


class HybridPredictor:
    """The LLM when it answers in time, otherwise the instant word-count engine."""

    def __init__(self, llm, ngram, timeout=0.6):
        self.llm, self.ngram, self.timeout = llm, ngram, timeout
        self.last = "ngram"

    def predict(self, text, n=3, cancelled=None):
        if self.llm.proc.is_alive():
            words = self.llm.predict(text, n, cancelled=cancelled, timeout=self.timeout)
            if words:
                self.last = "llm"
                return words
        self.last = "ngram"
        return self.ngram.predict(text, n)


# ---------------------------------------------------------------- spelling correction
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def edits1(w):
    """Every string one edit away: delete, swap neighbours, replace, insert (Norvig's corrector)."""
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    swaps = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in LETTERS]
    inserts = [a + c + b for a, b in splits for c in LETTERS]
    return set(deletes + swaps + replaces + inserts)


class Speller:
    """Fixes clear one-letter typos using the word-frequency list. Conservative on purpose:
    lowercase words of 3+ letters only (no names, ACRONYMS or numbers), and words you undo are
    remembered in user_words.txt and never corrected again."""
    COMMON = 40_000     # words more common than this rank are never "corrected"
    RARE_TO = 3_000     # a rare-but-real word is only swapped for a very common one

    def __init__(self, ngram, folder):
        self.ng = ngram
        self.rank = {w: i for i, w in enumerate(ngram.words)}
        self.path = os.path.join(folder, "user_words.txt")
        self.user = set()
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.user = {line.strip().lower() for line in f if line.strip()}

    def correct(self, word, prev=None):
        if len(word) < 3 or not word.isalpha() or not word.islower():
            return None
        if word in self.user:
            return None
        r = self.rank.get(word)
        if r is not None and r < self.COMMON:
            return None                                 # a normal word
        cands = [c for c in edits1(word) if c in self.rank]
        if r is not None:
            cands = [c for c in cands if self.rank[c] < self.RARE_TO]
        if not cands:
            return None
        follows = set(self.ng.bi.get(prev, [])) if prev else set()
        return min(cands, key=lambda c: (c not in follows, self.rank[c]))   # fits context, then most common

    def remember(self, word):
        word = word.lower()
        if word and word not in self.user:
            self.user.add(word)
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(word + "\n")
            except OSError:
                pass
