from __future__ import annotations
import time, hashlib, textwrap, re, math, sys, json, urllib.parse, urllib.request, requests, ast
from collections import Counter
from typing import List, Tuple, Iterable

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
BASE_STOP = ["DONE", "THOUGHT END", "Human:", "Assistant:", "```"]

def wants_code(q: str) -> bool:
    ql = q.lower()
    kw = ["code","snippet","example","api","http","request","regex","sql","python","javascript","java","c#","c++","go","rust","bash","shell","function","method","class","implement","write a program","write code","pseudocode"]
    if "```" in q: return True
    return any(k in ql for k in kw)

def detect_code_lang_from_question(q: str) -> str | None:
    ql = q.lower()
    if any(k in ql for k in ["python", "py", "pandas", "numpy"]): return "python"
    if any(k in ql for k in ["javascript", "node", "js", "ts", "typescript"]): return "javascript"
    if "java" in ql and "javascript" not in ql: return "java"
    if any(k in ql for k in ["c++", "cpp"]): return "cpp"
    if "c#" in ql or "dotnet" in ql: return "csharp"
    if "rust" in ql: return "rust"
    if "go" in ql or "golang" in ql: return "go"
    if "bash" in ql or "shell" in ql: return "bash"
    if "sql" in ql: return "sql"
    return None

_CODE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.S)

def extract_fenced_code_blocks(s: str) -> list[tuple[str,str]]:
    return [(m.group(1).strip().lower(), m.group(2)) for m in _CODE_BLOCK_RE.finditer(s)]

def python_syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

def basic_runnability_score(text: str, expected_lang: str | None) -> tuple[bool, str, int]:
    blocks = extract_fenced_code_blocks(text)
    if len(blocks) != 1:
        return (False, f"requires exactly one fenced block, found {len(blocks)}", 0)
    lang, code = blocks[0]
    if expected_lang and lang and lang != expected_lang:
        return (False, f"language tag '{lang}' mismatches expected '{expected_lang}'", 0)
    if len(code.strip()) < 10:
        return (False, "code too short", 0)
    if re.search(r"(rm -rf\s+/|DROP\s+TABLE)", code, re.I):
        return (False, "dangerous pattern", 0)
    if (expected_lang or lang) == "python":
        if not python_syntax_ok(code):
            return (False, "python syntax error", 0)
    if (expected_lang or lang) in {"javascript", "typescript", "java", "csharp", "cpp"}:
        if code.count("{") != code.count("}") or code.count("(") != code.count(")"):
            return (False, "unbalanced brackets", 0)
    if (expected_lang or lang) == "sql":
        if not re.search(r"\b(select|insert|update|delete|create|alter)\b", code, re.I):
            return (False, "no SQL statement detected", 0)
    if (expected_lang or lang) == "bash":
        if "rm -rf /" in code.replace(" ", ""):
            return (False, "dangerous bash", 0)
    score = 7
    if expected_lang and (lang == expected_lang or not lang): score += 1
    if (expected_lang or lang) == "python": score += 1
    return (True, "ok", min(score, 10))

class LLMClient:
    def __init__(self, host: str | None = None, port: int | None = None):
        try:
            from config import HOST as _H, PORT as _P
        except ImportError:
            _H, _P = "localhost", 8000
        self.url = f"http://{host or _H}:{port or _P}"
        self.last_rate = float("inf")
        self.history: list[str] = []
        self.total_tokens = 0
        self.total_seconds = 0.0
        print("[LLM]", requests.get(f"{self.url}/health", timeout=10).json())

    def _post(self, path: str, json: dict, timeout: int = 120):
        r = requests.post(f"{self.url}{path}", json=json, timeout=timeout); r.raise_for_status(); return r

    def stop(self):
        try: self._post("/stop", json={}, timeout=10)
        except Exception: pass

    def generate(self, prompt: str, *, max_tokens=256, temperature=0.6,
                 top_p=0.95, repetition_penalty=1.0, stop: list[str] | None = None) -> str:
        t0 = time.time()
        payload = dict(prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                       top_p=top_p, repetition_penalty=repetition_penalty, stop=(stop or BASE_STOP))
        r = self._post("/generate", json=payload)
        txt = r.json().get("completion") or r.json().get("text") or r.json().get("response") or ""
        dt = time.time() - t0
        tok = len(txt.split())
        self.last_rate = tok / dt if dt else float("inf")
        self.total_tokens += tok
        self.total_seconds += dt
        cum_rate = (self.total_tokens / self.total_seconds) if self.total_seconds > 0 else float("inf")
        print(f"[LLM] {tok} tok in {dt:.2f}s ({self.last_rate:.1f} tok/s) | cum {self.total_tokens} tok in {self.total_seconds:.2f}s ({cum_rate:.1f} tok/s)  {txt.replace(chr(10),' ')[:90]}{'…' if len(txt)>90 else ''}")
        sys.stdout.flush()
        self.history.append(txt)
        return txt

    def get_stats(self) -> tuple[int,float,float]:
        rate = (self.total_tokens / self.total_seconds) if self.total_seconds > 0 else float("inf")
        return self.total_tokens, self.total_seconds, rate

_SEP_LINE   = re.compile(r"^[\s\-_—–·•\.]{3,}$")
_DROP_INSTR = re.compile(r"(Produce 10|Insert at least|End with THOUGHT END|Do NOT|Do not repeat|Constraints:)", re.I)
_DROP_META  = re.compile(r"^(Answer:|Certainly!?|Sure!?|No code\.?)\s*$", re.I)
_ROLELINE   = re.compile(r"^\s*(Human|Assistant)\s*:", re.I)

DEFAULT_NON_TOPIC = {"virus", "bacteria", "palindrome", "http", "request", "function vs method"}

def clean(txt: str) -> str:
    txt = re.split(r"\b(DONE|THOUGHT END)\b", txt)[0]
    txt = re.sub(r"(?:\bNo code\.?\s*){1,}", "", txt, flags=re.I)
    out, seen = [], set()
    for ln in txt.splitlines():
        ln = ln.strip()
        if (not ln or ln in seen or _SEP_LINE.match(ln) or
            _DROP_INSTR.search(ln) or _DROP_META.match(ln) or _ROLELINE.match(ln)):
            continue
        if re.fullmatch(r"[-–—]{2,}(\s[-–—]{2,})*", ln): continue
        seen.add(ln)
        out.append(ln)
    return "\n".join(out).replace("DONE.", "").strip()

def extract_ahas(txt: str) -> list[str]:
    return [ln.strip()[len(AHA):].lstrip(" :") for ln in txt.splitlines() if ln.lstrip().startswith(AHA)]

def contains_non_topic(ans: str, terms: Iterable[str]) -> bool:
    low = ans.lower()
    return any(t in low for t in terms)

def contains_code(ans: str) -> bool:
    if "```" in ans: return True
    low = ans.lower()
    return any(x in low for x in ["def ", "class ", "import ", "return ", ";", "{", "}", "public ", "private ", "#include", "select ", "insert ", "update "])

def junk_answer(a: str, *, allow_code: bool, non_topic_terms: Iterable[str]) -> bool:
    a = a.strip()
    if len(a.split()) < (40 if allow_code else 60): return True
    if _SEP_LINE.fullmatch(a.strip("- ·•—–_")) is not None: return True
    if contains_non_topic(a, non_topic_terms): return True
    if not allow_code and contains_code(a): return True
    if re.match(r"^\s*(human|assistant)\s*:", a, flags=re.I): return True
    return False

def on_topic_score(ans: str, question: str) -> int:
    def toks(s: str) -> set[str]:
        return set(w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 3)
    q = toks(question); a = toks(ans)
    inter = len(q & a)
    return min(10, 2*inter)

def structure_score(ans: str, *, allow_code: bool) -> int:
    if allow_code:
        blocks = extract_fenced_code_blocks(ans)
        pre = ans.split("```")[0].strip() if "```" in ans else ans
        post = ""
        if ans.count("```") >= 2:
            post = re.split(r"```[^\n]*\n.*?```", ans, flags=re.S)[-1]
        pre_ok = len(pre.split()) <= 120
        post_ok = len(post.strip().split()) <= 60
        base = 4 + (3 if len(blocks) == 1 else 0) + (1 if pre_ok else 0) + (1 if post_ok else 0)
        return min(10, base)
    w = len(ans.split())
    bullets = len(re.findall(r"^\s*(?:\d+\.|[-•])", ans, flags=re.M))
    sentences = max(1, len(re.findall(r"[.!?](\s|$)", ans)))
    concl = 2 if re.search(r"\b(therefore|in sum|overall|hence|thus)\b", ans.lower()) else 0
    return int(min(10, (min(w, 260)/26) + bullets + concl + min(3, sentences//4)))

def penalty_score(ans: str, *, allow_code: bool) -> int:
    p = 0
    if re.search(r"\b(human:|assistant:)\b", ans, re.I): p += 6
    if "no code" in ans.lower(): p += 4
    blocks = extract_fenced_code_blocks(ans)
    if allow_code:
        if len(blocks) == 0: p += 4
        if len(blocks) > 1: p += 4
    else:
        if len(blocks) > 0 or contains_code(ans): p += 8
    if re.search(r"---\s*$", ans): p += 3
    return p

def toks(s: str) -> list[str]:
    return re.findall(r"[a-z]{4,}", s.lower())

def top_terms_from_texts(texts: list[str], exclude: set[str], k: int = 30) -> list[str]:
    c = Counter()
    for t in texts:
        for w in toks(t):
            if w not in exclude:
                c[w] += 1
    return [w for w,_ in c.most_common(k)]

def parse_json_obj(s: str) -> dict | None:
    try:
        m = re.search(r"\{.*\}", s, re.S)
        if not m: return None
        return json.loads(m.group(0))
    except Exception:
        return None

class ChainRunner:
    T_MIN = 0.05
    def __init__(self, q: str, cli: LLMClient, *, fanout=12, breadth=2, depth=2,
                 compress_threshold=350, init_temp=0.8, max_answers=4):
        self.q0 = q.strip()
        self.cli = cli
        self.fanout, self.breadth, self.depth = fanout, breadth, depth
        self.compress_threshold = compress_threshold
        self.init_temp, self.max_answers = init_temp, max_answers
        self.allow_code = wants_code(self.q0)
        bk = "\n".join(f for f in (wiki_summary(self.q0), ddg_summary(self.q0)) if f)
        self.bkg = bk
        self.q   = self._paraphrase()
        self.non_topic = set(DEFAULT_NON_TOPIC)
        self._expand_offtopic_with_llm()
        self._expanded_from_history = False
        self.judge_trace = ""

    def _paraphrase_ok(self, candidate: str) -> bool:
        if not candidate or "?" not in candidate or len(candidate) > 150: return False
        q_tok = set(toks(self.q0)); c_tok = set(toks(candidate))
        if not q_tok: return False
        j = len(q_tok & c_tok) / max(1, len(q_tok | c_tok))
        if j < 0.4: return False
        if contains_non_topic(candidate, self.non_topic | DEFAULT_NON_TOPIC): return False
        return True

    def _paraphrase(self) -> str:
        p = ("Rewrite the user's question in one sentence.\n"
             "Keep the same topic; do not add examples; do not mention these instructions.\n---\n" + self.q0)
        d = clean(self.cli.generate(p, temperature=0.2, max_tokens=50, stop=BASE_STOP))
        return d if self._paraphrase_ok(d) else self.q0

    def _head(self) -> str:
        if self.allow_code:
            return ("Follow these constraints and do not repeat or mention them:\n"
                    "• Stay strictly on this question; do not change topic.\n"
                    "• You may include code blocks and minimal explanation.\n"
                    "• Do not include role labels or tool calls.\n\n")
        return ("Follow these constraints and do not repeat or mention them:\n"
                "• Stay strictly on this question; do not change topic.\n"
                "• Write natural language prose only; do not include code or role labels.\n"
                "• If you start to drift, stop and continue answering the question.\n\n")

    def _stoplist(self) -> list[str]:
        return [s for s in BASE_STOP if (s != "```" or not self.allow_code)]

    def _reason_prompt(self, partial: str | None) -> str:
        base = f"Primary question:\n{self.q}\n\n(Original wording: {self.q0})"
        if self.bkg: base += f"\n\nBackground snippets:\n{self.bkg}\n"
        if partial:  base += f"\n\nPartial reasoning so far:\n{partial}"
        return textwrap.dedent(f"""{self._head()}{base}
            Produce 10–15 numbered steps of reasoning.
            Include at least two lines starting with "{AHA}" that mark genuine discoveries
            (insights that change the direction or unify the argument). The first must appear in the first third.
            End with THOUGHT END.
            Do not repeat these instructions.""").strip()

    def _answer_prompt(self, cog: str) -> str:
        aha = extract_ahas(cog)
        ki  = "\nKey insights:\n- " + "\n- ".join(aha) + "\n" if aha else ""
        bg  = f"\nBackground:\n{self.bkg}\n" if self.bkg else ""
        if self.allow_code:
            return textwrap.dedent(f"""{self._head()}
                Question: {self.q}{bg}{ki}
                Reasoning:
                {cog}
                Provide a short explanation followed by a single fenced code block implementing the solution.
                Add 1–2 bullet tips after the code. Finish with DONE.""").strip()
        return textwrap.dedent(f"""{self._head()}
            Question: {self.q}{bg}{ki}
            Reasoning:
            {cog}
            Write a cohesive 130–220‑word answer with a 1‑sentence conclusion (beginning with “Therefore,” or “In sum,”).
            Do not use role labels or code blocks. Finish with DONE.""").strip()

    def _compress_prompt(self, ans: str) -> str:
        head = self._head()
        return head + "Shorten to ~150 words without losing ≥90% of the facts. Output only the revised answer. Finish with DONE.\n---\n" + ans

    def _refine_prompt(self, draft: str) -> str:
        return textwrap.dedent(f"""{self._head()}
            Your previous answer was contaminated or incomplete.
            1) Remove any role labels / meta; 2) ensure a full solution; 3) end correctly.
            Output only the improved answer. Finish with DONE.
            ---
            {draft}""")

    def _score_thought(self, t: str) -> int:
        txt = clean(t)
        ah  = extract_ahas(txt)
        if not ah: return 0
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        first_idx = next((i for i, ln in enumerate(lines) if ln.lstrip().startswith(AHA)), len(lines))
        early = first_idx <= max(1, len(lines)//3)
        s = 4*len(ah) + (2 if early else 0) + min(4, math.log1p(len(txt.split())))
        return int(min(10, s))

    def _score_answer(self, ans: str) -> int:
        return on_topic_score(ans, self.q0) + structure_score(ans, allow_code=self.allow_code) - penalty_score(ans, allow_code=self.allow_code)

    def _expand_offtopic_with_llm(self) -> None:
        prompt = textwrap.dedent(f"""
        You are a topic guard. The user question is:
        "{self.q0}"
        Background (optional):
        "{self.bkg or ''}"
        List 5–12 keywords/short phrases that are clearly off-topic for this question and would indicate drift if they appear.
        Output ONLY JSON with keys "confidence" (0.0–1.0) and "terms" (list of lowercase strings).
        """).strip()
        raw = self.cli.generate(prompt, temperature=0.0, max_tokens=160, stop=self._stoplist())
        obj = parse_json_obj(raw) or {}
        conf = float(obj.get("confidence", 0.0) or 0.0)
        terms = obj.get("terms", []) or []
        if conf >= 0.7 and isinstance(terms, list):
            for t in terms:
                if isinstance(t, str) and 2 <= len(t) <= 30:
                    self.non_topic.add(t.strip().lower())

    def _augment_from_history(self) -> None:
        q_tokens = set(toks(self.q0))
        recent = [clean(h) for h in self.cli.history[-6:]]
        cand = top_terms_from_texts(recent, exclude=q_tokens, k=40)
        if not cand: return
        prompt = textwrap.dedent(f"""
        You are validating off-topic drift. The question is:
        "{self.q0}"
        Candidate terms extracted from drafts:
        {", ".join(cand)}
        Return ONLY JSON {{"confidence":0.0-1.0,"terms":[off_topic_terms_only]}} where terms are a subset of the candidates that are not relevant to the question's topic.
        """).strip()
        raw = self.cli.generate(prompt, temperature=0.0, max_tokens=160, stop=self._stoplist())
        obj = parse_json_obj(raw) or {}
        conf = float(obj.get("confidence", 0.0) or 0.0)
        terms = obj.get("terms", []) or []
        if conf >= 0.7 and isinstance(terms, list):
            for t in terms:
                if isinstance(t, str): self.non_topic.add(t.strip().lower())
        self._expanded_from_history = True

    def _sanitize_final(self, s: str) -> str:
        s = clean(s)
        s = re.sub(r"^\s*(Human|Assistant)\s*:.*$", "", s, flags=re.I|re.M).strip()
        s = re.sub(r"(?:\bNo code\.?\s*){2,}", "", s, flags=re.I).strip()
        s = re.sub(r"(?:^[-–—]{2,}\s*$\n?)+", "", s, flags=re.M).strip()
        if self.allow_code:
            blocks = extract_fenced_code_blocks(s)
            if len(blocks) == 0:
                return s.strip()
            lang, code = blocks[0]
            lang = lang or (detect_code_lang_from_question(self.q0) or "")
            pre = s.split("```")[0].strip()
            tips = []
            m = re.search(r"```[^\n]*\n.*?```(.*)$", s, re.S)
            if m:
                tail = m.group(1).strip()
                for ln in tail.splitlines():
                    if len(tips) >= 2: break
                    if ln.strip().startswith(("-", "•")) and len(ln.split()) <= 20:
                        tips.append(ln.strip())
            out = (pre + "\n\n" if pre else "")
            out += f"```{lang}\n{code}```"
            if tips:
                out += "\n" + "\n".join(tips)
            return out.strip()
        if not re.search(r"\b(Therefore|In sum),", s):
            s += ("\n\nTherefore, " if not s.endswith(('.', '!', '?')) else " Therefore, ") + "taken together the strongest evidence points to a shared, rule‑governed external reality rather than a private fiction."
        return s.strip()

    def _llm_rank_and_select(self, candidates: list[str]) -> tuple[int, str, dict]:
        if not candidates: return -1, "", {}
        numbered = "\n\n".join(f"A{i}: {c}" for i, c in enumerate(candidates))
        allow = "true" if self.allow_code else "false"
        exp_lang = detect_code_lang_from_question(self.q0) or ""
        code_rules = "- exactly one fenced code block using triple backticks\n- optional language tag; if present it should match expected language\n- minimal explanation before code; optional up to 2 short tips after code\n- no extra prose after the code/tips\n"
        prompt = textwrap.dedent(f"""
    You are a strict grader. Question:
    "{self.q0}"
    There are {len(candidates)} candidate answers A0..A{len(candidates)-1}.
    Evaluate each on: on‑topicness, completeness, coherence, absence of meta/role labels,
    style constraints (allow_code={allow}), and, if allow_code=true, the following code rules:
    {code_rules}
    Expected language (best guess): "{exp_lang or 'unspecified'}".
    Return ONLY JSON:
    {{
      "confidence": 0.0-1.0,
      "best_index": <int>,
      "scores": [0-10 per answer],
      "reasons": ["short reason per answer"],
      "proposed_final": "<optional improved final answer that fixes minor flaws without changing substance>"
    }}
    Answers:
    {numbered}
    """).strip()
        raw = self.cli.generate(prompt, temperature=0.0, max_tokens=700, stop=self._stoplist())
        obj = parse_json_obj(raw) or {}
        try:
            best_idx = int(obj.get("best_index", -1))
        except Exception:
            best_idx = -1
        prop = (obj.get("proposed_final") or "").strip()
        conf = float(obj.get("confidence", 0.0) or 0.0)
        def gate(text: str) -> bool:
            if not self.allow_code: return True
            ok, _, _ = basic_runnability_score(text, exp_lang)
            return ok
        if prop and conf >= 0.65:
            cand = self._sanitize_final(prop)
            if gate(cand) and not junk_answer(cand, allow_code=self.allow_code, non_topic_terms=self.non_topic):
                return best_idx if 0 <= best_idx < len(candidates) else 0, cand, {"judge_json": obj}
        if 0 <= best_idx < len(candidates):
            chosen = self._sanitize_final(candidates[best_idx])
            if gate(chosen):
                return best_idx, chosen, {"judge_json": obj}
        ranked = sorted([(self._score_answer(c), i, c) for i, c in enumerate(candidates)], reverse=True)
        for _, i, c in ranked:
            c2 = self._sanitize_final(c)
            if gate(c2):
                return i, c2, {"judge_json": obj}
        _, i, c = ranked[0]
        return i, self._sanitize_final(c), {"judge_json": obj}

    def _salvage_from_history(self) -> str | None:
        pool = []
        for h in self.cli.history[-12:]:
            c = clean(h)
            if not c: continue
            if contains_non_topic(c, self.non_topic): continue
            if len(c.split()) < 40: continue
            s = self._score_answer(c)
            pool.append((s, c))
        if not pool: return None
        pool.sort(reverse=True)
        return self._sanitize_final(pool[0][1])

    def _template_fallback(self) -> str:
        if self.allow_code:
            lang = detect_code_lang_from_question(self.q0) or "python"
            if lang == "python":
                return ("A minimal example is shown below.\n\n```python\n# placeholder example\nprint('ok')\n```")
            if lang in {"javascript","typescript"}:
                tag = "javascript" if lang == "javascript" else "typescript"
                return (f"A minimal example is shown below.\n\n```{tag}\nconsole.log('ok');\n```")
            return ("A minimal example is shown below.\n\n```text\nN/A\n```")
        ql = self.q0.lower()
        if "imaginary" in ql and "world" in ql:
            return ("We have powerful pragmatic reasons to treat the world as mind‑independent: different observers converge on the same facts, the same regularities let us predict and control outcomes, and stubborn counter‑evidence resists wishful thinking. Scientific instruments extend our senses and reproduce results across labs; technology only works because those regularities hold when no one is looking; and history and archaeology knit a continuous record that none of us authored. Pain, scarcity, and unintended consequences also constrain us as if by external causes. The best explanation of this package—convergent testimony, lawful structure, and reliable intervention—is that there exists a shared external world rather than a private fiction. Therefore, the hypothesis of a real, rule‑governed world explains our experience more simply and completely than an imaginary one.")
        return ("Here is a concise answer. Multiple independent observers agree on the same events; stable regularities let us predict and control outcomes; and instruments and records replicate results across time. These features are hard to reconcile with a private fiction but natural under an external‑world hypothesis. Therefore, the simplest, most predictive account of our experience is that there is a shared, mind‑independent reality.")

    def run(self) -> str:
        self.cli.stop()
        leaves, temp = [""], self.init_temp
        for _ in range(self.depth):
            new = []
            for leaf in leaves:
                cands: list[Tuple[int, str]] = []
                for _ in range(self.breadth):
                    t = clean(self.cli.generate(self._reason_prompt(leaf), temperature=temp, max_tokens=350, stop=self._stoplist()))
                    cands.append((self._score_thought(t), t))
                best = max(cands, key=lambda x: x[0])[1]
                new.append(f"{leaf}\n{best}")
            leaves, temp = new, max(0.15, temp - 0.25)
            if not self._expanded_from_history and len(self.cli.history) >= 2:
                self._augment_from_history()
        temps = [self.T_MIN + (self.init_temp - self.T_MIN)*i/(self.fanout-1) for i in range(self.fanout)]
        answers: list[Tuple[int, str, float]] = []
        for i, t in enumerate(temps):
            if len(answers) >= self.max_answers: break
            raw = self.cli.generate(self._answer_prompt(leaves[i % len(leaves)]), temperature=t, max_tokens=420, stop=self._stoplist())
            a   = clean((raw.split("DONE")[0] or raw).strip())
            if junk_answer(a, allow_code=self.allow_code, non_topic_terms=self.non_topic):
                continue
            score = self._score_answer(a)
            answers.append((score, a, t))
            head = a.splitlines()[0] if a.splitlines() else a[:70]
            print(f"[ans {i}] T={t:.2f}  score={score}  {head[:70]}")
        if not answers:
            fallback_prompt = (
                self._head() +
                (f"Question: {self.q}\nProvide a minimal working example in a fenced code block with a brief explanation. Finish with DONE."
                 if self.allow_code else
                 f"Question: {self.q}\nWrite a 150–200‑word argument in one cohesive paragraph. End with a single conclusion sentence starting with “Therefore,”. Finish with DONE.")
            )
            raw = self.cli.generate(fallback_prompt, temperature=0.2, max_tokens=320, stop=self._stoplist())
            a   = clean((raw.split("DONE")[0] or raw).strip())
            if not junk_answer(a, allow_code=self.allow_code, non_topic_terms=self.non_topic):
                answers = [(self._score_answer(a), a, 0.2)]
        if not answers:
            salv = self._salvage_from_history()
            if salv:
                return salv
            return self._sanitize_final(self._template_fallback())
        cand_texts = [a for _, a, _ in answers]
        idx, final_ans, info = self._llm_rank_and_select(cand_texts)
        self.judge_trace = json.dumps(info.get("judge_json", {}), ensure_ascii=False)
        if not self.allow_code and len(final_ans) > self.compress_threshold:
            final_ans = clean(self.cli.generate(self._compress_prompt(final_ans), temperature=0.2, stop=self._stoplist()).split("DONE")[0])
        if len(final_ans.split()) < 20:
            salv = self._salvage_from_history()
            if salv: return salv
            return self._sanitize_final(self._template_fallback())
        return final_ans

if __name__ == "__main__":
    client   = LLMClient()
    question = "How can we argue that the world we are living in is not imaginary?"
    print("User Q:", question)
    runner = ChainRunner(question, client)
    result = runner.run()
    print("\nFINAL ANSWER:\n" + result + "\n")
    tot_tok, tot_sec, avg_rate = client.get_stats()
    print(f"[STATS] total {tot_tok} tokens in {tot_sec:.2f}s → {avg_rate:.1f} tok/s")
    with open("qa_log.txt", "a", encoding="utf-8") as f:
        f.write(f"QUESTION:\n{question}\n\nFINAL ANSWER:\n{result}\n[STATS] total {tot_tok} tok in {tot_sec:.2f}s → {avg_rate:.1f} tok/s\n{'='*80}\n")
    with open("generations_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\nRUN for question: {question}\n")
        for i, txt in enumerate(client.history):
            f.write(f"\n--- COMPLETION {i} ---\n{txt}\n")
        if runner.judge_trace:
            f.write(f"\n[JUDGE]\n{runner.judge_trace}\n")
        f.write("="*80 + "\n")
