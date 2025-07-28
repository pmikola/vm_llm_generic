from __future__ import annotations
import time, hashlib, textwrap, re, math, sys, json, urllib.parse, urllib.request, requests
from collections import Counter
from typing import List, Tuple

def simhash64(s: str) -> str:
    return hashlib.blake2b(s.encode(), digest_size=8).hexdigest()

def maj_vote(items: list[str]) -> str:
    return Counter(items).most_common(1)[0][0]

def wiki_summary(topic: str, n: int = 2) -> str:
    try:
        u = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic)}"
        d = json.loads(urllib.request.urlopen(u, timeout=5).read().decode())
        return ". ".join(d.get("extract", "").split(". ")[:n])
    except Exception:
        return ""

def ddg_summary(q: str, n: int = 2) -> str:
    try:
        u = f"https://api.duckduckgo.com/?q={urllib.parse.quote(q)}&format=json&no_redirect=1&no_html=1"
        d = json.loads(urllib.request.urlopen(u, timeout=5).read().decode())
        return ". ".join(d.get("AbstractText", "").split(". ")[:n])
    except Exception:
        return ""

AHA = "💡 AHA"
STOP = ["DONE", "THOUGHT END"]

class LLMClient:
    def __init__(self, host: str | None = None, port: int | None = None):
        try:
            from config import HOST as _H, PORT as _P
        except ImportError:
            _H, _P = "localhost", 8000
        self.url = f"http://{host or _H}:{port or _P}"
        self.last_rate = float("inf")
        self.history: list[str] = []
        print("[LLM]", requests.get(f"{self.url}/health", timeout=10).json())

    def _post(self, path: str, json: dict, timeout: int = 120):
        r = requests.post(f"{self.url}{path}", json=json, timeout=timeout); r.raise_for_status(); return r

    def stop(self):
        try: self._post("/stop", json={}, timeout=10)
        except Exception: pass

    def generate(self, prompt: str, *, max_tokens=256, temperature=0.6, top_p=0.95, repetition_penalty=1.0) -> str:
        t0 = time.time()
        r = self._post("/generate", json=dict(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, stop=STOP))
        txt = r.json().get("completion") or r.json().get("text") or r.json().get("response") or ""
        dt = time.time() - t0
        self.last_rate = len(txt.split()) / dt if dt else float("inf")
        print(f"[LLM] {len(txt.split())} tok {dt:.2f}s ({self.last_rate:.1f} tok/s)  {txt.replace(chr(10),' ')[:90]}{'…' if len(txt)>90 else ''}")
        sys.stdout.flush()
        self.history.append(txt)
        return txt

DROP = re.compile(r"(Produce 10|Insert at least|THOUGHT END)", re.I)

def clean(t: str) -> str:
    t = re.split(r"\b(DONE|THOUGHT END)\b", t)[0]
    seen, out = set(), []
    for ln in t.splitlines():
        ln = ln.strip()
        if not ln or ln in seen or DROP.search(ln) or ln.startswith(("Human:", "Assistant:")):
            continue
        seen.add(ln)
        out.append(ln)
    return "\n".join(out)

def extract_ahas(txt: str) -> list[str]:
    return [ln.strip()[len(AHA):].lstrip(" :") for ln in txt.splitlines() if ln.lstrip().startswith(AHA)]

class ChainRunner:
    T_MIN = 0.05
    def __init__(self, q: str, cli: LLMClient, *, fanout=12, breadth=2, depth=2, compress_threshold=350, init_temp=0.8, max_answers=4):
        self.q0 = q.strip()
        self.cli = cli
        self.fanout, self.breadth, self.depth = fanout, breadth, depth
        self.compress_threshold = compress_threshold
        self.init_temp, self.max_answers = init_temp, max_answers
        bk = "\n".join(f for f in (wiki_summary(self.q0), ddg_summary(self.q0)) if f)
        self.bkg = bk
        self.q = self._paraphrase()

    def _paraphrase(self) -> str:
        p = "Paraphrase in one sentence.\n---\n" + self.q0
        d = clean(self.cli.generate(p, temperature=0.2, max_tokens=50))
        return d if "?" in d and len(d) < 150 else self.q0

    def _prompt_head(self) -> str: return "No code.\n\n"

    def _reason_prompt(self, partial: str | None) -> str:
        base = f"{self.q}\n\n({self.q0})"
        if self.bkg: base += f"\n\nBackground:\n{self.bkg}\n"
        if partial: base += f"\n\nPartial reasoning:\n{partial}"
        return textwrap.dedent(f"""{self._prompt_head()}{base}
            Produce 10‑15 numbered steps.
            Insert at least TWO "{AHA}" lines (discoveries). First within first third.
            End with THOUGHT END.""").strip()

    def _answer_prompt(self, cog: str) -> str:
        ah = extract_ahas(cog)
        ki = "\nKey insights:\n- " + "\n- ".join(ah) + "\n" if ah else ""
        bg = f"\nBackground:\n{self.bkg}\n" if self.bkg else ""
        return textwrap.dedent(f"""{self._prompt_head()}
            Question: {self.q}{bg}{ki}
            Reasoning:
            {cog}
            Provide a clear answer. Finish with DONE.""").strip()

    def _compress_prompt(self, ans: str) -> str:
        return f"{self._prompt_head()}Shorten (keep ≥90 % facts). Return then DONE.\n---\n{ans}"

    def _refine_prompt(self, draft: str) -> str:
        return textwrap.dedent(f"""{self._prompt_head()}
            List any errors or gaps, then give an improved answer only.
            Output just the improved answer.
            ---
            {draft}""")

    def _score(self, thought: str) -> int:
        txt = clean(thought)
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        pos = [i for i, ln in enumerate(lines) if ln.lstrip().startswith(AHA)]
        if not pos: return 0
        early = pos[0] <= max(1, len(lines)//3)
        return int(min(10, 4*len(pos) + (2 if early else 0) + min(4, math.log1p(len(txt.split())))))

    def run(self) -> str:
        self.cli.stop()
        leaves, temp = [""], self.init_temp
        for _ in range(self.depth):
            new = []
            for leaf in leaves:
                scored = []
                for _ in range(self.breadth):
                    t = clean(self.cli.generate(self._reason_prompt(leaf), temperature=temp, max_tokens=350))
                    scored.append((self._score(t), t))
                new.append(f"{leaf}\n{max(scored)[1]}")
            leaves, temp = new, max(0.15, temp-0.25)
        temps = [self.T_MIN + (self.init_temp-self.T_MIN)*i/(self.fanout-1) for i in range(self.fanout)]
        ans, at = [], []
        for i, t in enumerate(temps):
            if len(ans) >= self.max_answers: break
            raw = self.cli.generate(self._answer_prompt(leaves[i % len(leaves)]), temperature=t, max_tokens=400)
            a = clean(raw.split("DONE")[0] or raw)
            ans.append(a); at.append(t)
            print(f"[ans {i}] T={t:.2f}  {a.splitlines()[0][:70]}")
        best = maj_vote(ans); bt = at[ans.index(best)]
        ref = clean(self.cli.generate(self._refine_prompt(best), temperature=0.3, max_tokens=350))
        if len(ref.split()) > 20: best = ref
        if len(best) > self.compress_threshold:
            best = clean(self.cli.generate(self._compress_prompt(best), temperature=0.2).split("DONE")[0])
        return f"Best answer (T={bt:.2f}):\n{best}\n\nDONE"

if __name__ == "__main__":
    client = LLMClient()
    q = "How can we argue that the world we are living in is not imaginary?"
    print("User Q:", q)
    res = ChainRunner(q, client).run()
    print(res)
    with open("qa_log.txt","a",encoding="utf-8") as f: f.write(f"Q:\n{q}\n\nA:\n{res}\n{'='*80}\n")
    with open("generations_log.txt","a",encoding="utf-8") as f:
        f.write(f"\nRUN for: {q}\n"); [f.write(f"\n--- COMPLETION {i}\n{t}\n") for i,t in enumerate(client.history)]
        f.write("="*80+"\n")
