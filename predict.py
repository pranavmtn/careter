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

    def __init__(self, name="HuggingFaceTB/SmolLM2-135M", threads=2, max_ctx_tokens=96):
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
        """[(token_id, logprob)] for the k most likely next tokens after ids."""
        ids = (self.bos + ids)[-self.max_ctx:] or self.encode("\n")
        torch = self.torch
        with torch.inference_mode():
            logits = self.model(torch.tensor([ids])).logits[0, -1].float()
            logp = torch.log_softmax(logits, -1)
            v, i = torch.topk(logp, k)
        return list(zip(i.tolist(), v.tolist()))


class Predictor:
    def __init__(self, lm, seeds=40, max_word_tokens=5):
        self.lm, self.seeds, self.max_word_tokens = lm, seeds, max_word_tokens
        self.lock = threading.Lock()

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
            cands = self._seeds(ids, lead, p)
            words = {}
            for tok_ids, word, done, score in cands:
                word, score = self._extend(ids, tok_ids, word, done, score, p)
                if word and len(word) > len(partial) and word.lower().startswith(p):
                    key = word.lower()
                    if key not in words or words[key][1] < score:
                        words[key] = (word, score)
                if len(words) >= n:
                    break
        ranked = sorted(words.values(), key=lambda ws: -ws[1])[:n]
        out = []
        for word, _ in ranked:
            shown = partial + word[len(partial):]       # keep the user's own capitalisation
            out.append((shown, word[len(partial):] + " "))
        return out

    def _consistent(self, word, p):
        w = word.lower()
        return w.startswith(p) or p.startswith(w)

    def _seeds(self, ids, lead, p):
        """Top next tokens that start a word consistent with what's typed: (ids, word, done, logprob)."""
        out = []
        for tid, lp in self.lm.next_top(ids, self.seeds * 5):
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
            if not self._consistent(word, p) or (done and len(word) < len(p)):
                continue
            out.append(([tid], word, done, lp))
            if len(out) >= self.seeds:
                break
        return out

    def _extend(self, ids, tok_ids, word, done, score, p):
        """Greedily add tokens until the word ends; None if it can't match the typed part."""
        for _ in range(self.max_word_tokens):
            if done:
                break
            nxt = None
            for tid, lp in self.lm.next_top(ids + tok_ids, 20):
                t = self.lm.vocab[tid] if tid < len(self.lm.vocab) else None
                if not t:
                    continue
                m = LEAD_RE.match(t)
                if not m:                               # space or punctuation: the word ends here
                    if len(word) >= len(p):
                        nxt = ("end", None, None, None)
                        break
                    continue                            # too early, the user typed more than this
                piece = m.group(0)
                if self._consistent(word + piece, p):
                    nxt = ("add", tid, piece, lp, len(piece) < len(t))
                    break
            if nxt is None:
                return None, score
            if nxt[0] == "end":
                break
            _, tid, piece, lp, ends = nxt
            tok_ids, word, score, done = tok_ids + [tid], word + piece, score + lp, ends
        return word, score
