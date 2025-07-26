from __future__ import annotations

import os
import textwrap
from typing import List

import requests


class LLMClient:
    def __init__(self, host: str | None = None, port: int | None = None):
        try:
            from config import HOST as _H, PORT as _P
        except Exception:
            _H, _P = "localhost", 8000
        self.host = host or _H
        self.port = port or _P
        self.url = f"http://{self.host}:{self.port}"
        print("[LLM] Health status:", requests.get(f"{self.url}/health", timeout=10).json())

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 300,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> str:
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
        r = requests.post(f"{self.url}/generate", json=payload, timeout=120)
        r.raise_for_status()
        d = r.json()
        return d.get("completion") or d.get("text") or d.get("response") or ""


class ChainOfThoughtRunner:
    def __init__(
        self,
        question: str,
        client: LLMClient,
        *,
        temps: List[float] | None = None,
        max_passes: int = 3,
        compress_threshold: int = 1500,
    ):
        self.question = question.strip()
        self.client = client
        self.temps = temps or [0.2, 0.5, 0.8]
        self.max_passes = max_passes
        self.compress_threshold = compress_threshold

    def _prompt_reason(self) -> str:
        return textwrap.dedent(
            f"""
            {self.question}

            Let's think step by step. Solve the problem thoroughly but do not output the final answer yet—only your detailed reasoning.
            """
        ).strip()

    def _prompt_select(self, cots: List[str]) -> str:
        joined = "\n\n".join(f"### Reasoning #{i+1}\n{cot}" for i, cot in enumerate(cots))
        return textwrap.dedent(
            f"""
            You are an expert at evaluating reasoning quality.
            Below are several chains of thought produced for the same question. Identify which reasoning is the most coherent, correct, and complete.

            {joined}

            Reply only with the number of the best reasoning (e.g. 2). If none are acceptable, reply with 0.
            """
        ).strip()

    def _prompt_answer(self, reasoning: str) -> str:
        return textwrap.dedent(
            f"""
            Question: {self.question}

            The following chain of thought appears to be the best:
            {reasoning}

            Please craft a concise, direct answer based on that reasoning. When you believe the answer is complete, append the token DONE on its own line.
            """
        ).strip()

    def _prompt_compress(self, answer: str) -> str:
        return textwrap.dedent(
            f"""
            The answer below is too long (>{self.compress_threshold} characters). Compress it so it remains accurate and clear but shorter. Do not omit key details. End your response with the token DONE on its own line.

            ---
            {answer}
            """
        ).strip()

    def run(self) -> str:
        for p in range(1, self.max_passes + 1):
            print(f"\n[Pass {p}/{self.max_passes}] Generating chains of thought …")
            cots = [self.client.generate(self._prompt_reason(), temperature=t).strip() for t in self.temps]
            choice_raw = self.client.generate(
                self._prompt_select(cots),
                temperature=0.0,
                top_p=0.1,
            ).strip()
            print("   Model's selection string:", repr(choice_raw))
            try:
                idx = int(choice_raw.split()[0]) - 1
            except Exception:
                idx = 0
            if idx not in range(len(cots)):
                idx = 0
            best_cot = cots[idx]
            answer = self.client.generate(self._prompt_answer(best_cot), temperature=0.2).strip()
            if "DONE" in answer.upper():
                print("   ✔ Model signalled completion.")
                if len(answer) > self.compress_threshold:
                    print("   ↪ Compressing answer …")
                    answer = self.client.generate(self._prompt_compress(answer), temperature=0.2).strip()
                return answer
            print("   ↪ Model did not finish; refining question and retrying …")
            self.question = textwrap.dedent(
                f"""
                {self.question}

                Earlier attempt (may be flawed):
                {answer}

                Please refine your reasoning and solution.
                """
            ).strip()
        return "Model failed to produce a final answer within the allotted passes."


if __name__ == "__main__":
    HOST = os.getenv("HOST") or None
    PORT = int(os.getenv("PORT") or 0) or None
    client = LLMClient(host=HOST, port=PORT)
    QUESTION = "What are three compelling arguments that prove the Apollo Moon landings were real?"
    runner = ChainOfThoughtRunner(
        QUESTION,
        client,
        temps=[0.2, 0.5, 0.8],
        max_passes=3,
        compress_threshold=1200,
    )
    final = runner.run()
    print("\n================ FINAL ANSWER ================\n")
    print(final)
