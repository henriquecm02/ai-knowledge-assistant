"""Recuperação híbrida: BM25 (léxico) + embeddings (semântico), combinados por Reciprocal Rank Fusion.

- BM25 acerta termos exatos (valores, siglas, nomes de sistemas) e não precisa de modelo.
- Embeddings acertam paráfrases ("posso vender meus dias de folga?" vs. "banco de horas").
- RRF combina rankings sem precisar calibrar escalas de score diferentes.
O índice vetorial é uma matriz numpy: para bases de até dezenas de milhares de trechos, busca
exata por produto interno é rápida e dispensa um banco vetorial.
"""
from __future__ import annotations

import math
import re
import unicodedata
import zlib
from collections import Counter
from typing import Protocol

import numpy as np

from .ingest import Chunk

STOP = set("""a o e é de do da dos das em no na nos nas um uma uns umas por para com sem que se ao aos à às
ou como mais menos quando onde qual quais quem ser ter há foi são está estão pelo pela pelos pelas
eu você meu minha seu sua isso isto esse essa este esta lhe nos eles elas me te até após sobre entre
posso pode preciso devo tenho quanto quantos quantas qual""".split())


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def stem(t: str) -> str:
    """Stemming leve para português: remove plurais e algumas terminações comuns."""
    for suf in ("mente", "coes", "cao", "oes", "ais", "eis", "res", "ns", "s"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)] + ("ca" if suf in ("coes", "cao") else "")
    return t


def tokenize(s: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+(?:[.,]\d+)*", fold(s)) if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.4, b: float = 0.75):
        self.toks = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.toks)) / max(len(self.toks), 1)
        df = Counter(t for d in self.toks for t in set(d))
        n = len(self.toks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.toks]

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.toks))
        for i, (tf, toks) in enumerate(zip(self.tf, self.toks)):
            norm = self.k1 * (1 - self.b + self.b * len(toks) / self.avg)
            out[i] = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf)
        return out


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str], kind: str) -> np.ndarray: ...  # kind: "documento" | "consulta"


class GeminiEmbedder:
    """Embeddings do Gemini com cache por texto e espera quando o limite por minuto da cota gratuita estoura."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001", batch: int = 50, max_wait: float = 90.0):
        from google import genai
        self.client, self.model, self.batch, self.name = genai.Client(api_key=api_key), model, batch, model
        self.max_wait = max_wait
        self._cache: dict[tuple[str, str], list[float]] = {}  # (tarefa, texto) -> vetor: a mesma pergunta não é paga duas vezes

    def _embed_batch(self, texts: list[str], task: str):
        import re
        import time

        from google.genai import errors, types
        for attempt in range(4):
            try:
                return self.client.models.embed_content(model=self.model, contents=texts,
                                                        config=types.EmbedContentConfig(task_type=task))
            except errors.APIError as e:
                m = re.search(r"retry in ([\d.]+)s", str(e))
                wait = float(m.group(1)) + 1 if m else 0.0
                # só vale esperar o limite POR MINUTO; o limite diário não se resolve esperando
                if getattr(e, "code", 0) != 429 or "PerDay" in str(e) or not wait or wait > self.max_wait or attempt == 3:
                    raise
                time.sleep(wait)

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        task = "RETRIEVAL_DOCUMENT" if kind == "documento" else "RETRIEVAL_QUERY"
        missing = [t for t in dict.fromkeys(texts) if (task, t) not in self._cache]
        for i in range(0, len(missing), self.batch):
            part = missing[i:i + self.batch]
            r = self._embed_batch(part, task)
            for t, e in zip(part, r.embeddings):
                self._cache[(task, t)] = e.values
        m = np.array([self._cache[(task, t)] for t in texts], dtype=np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)


class HashingEmbedder:
    """Embedding determinístico por hashing de tokens e bigramas de caracteres. NÃO é semântico:
    existe para testes offline e para o modo sem chave. Mede a infraestrutura, não a qualidade."""
    name = "hashing-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                m[i, zlib.crc32(b"w" + tok.encode()) % self.dim] += 1.0
                for a, b in zip(tok, tok[1:]):
                    m[i, zlib.crc32(("c" + a + b).encode()) % self.dim] += 0.3
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(n == 0, 1, n)


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    agg: dict[int, float] = {}
    for r in rankings:
        for pos, idx in enumerate(r):
            agg[idx] = agg.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return sorted(agg.items(), key=lambda x: -x[1])


class Index:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        self.chunks = chunks
        self.bm25 = BM25([f"{c.section}\n{c.text}" for c in chunks])
        self.embedder = embedder
        self.vectors = embedder.embed([f"{c.doc} · {c.section}\n{c.text}" for c in chunks], "documento") \
            if embedder else None

    def search(self, query: str, k: int = 5, mode: str = "hibrido", pool: int = 20) -> list[dict]:
        """mode: "bm25" | "denso" | "hibrido". Retorna trechos com scores de cada etapa."""
        bm = self.bm25.scores(query)
        bm_rank = [int(i) for i in np.argsort(-bm)[:pool] if bm[i] > 0]
        rankings, dense = [], None
        if mode in ("bm25", "hibrido"):
            rankings.append(bm_rank)
        if mode in ("denso", "hibrido") and self.vectors is not None:
            q = self.embedder.embed([query], "consulta")[0]
            dense = self.vectors @ q
            rankings.append([int(i) for i in np.argsort(-dense)[:pool]])
        if mode == "denso" and self.vectors is None:
            raise ValueError("Modo denso exige um embedder.")
        fused = rrf(rankings) if len(rankings) > 1 else [(i, 1.0 / (61 + p)) for p, i in enumerate(rankings[0])]
        out = []
        for idx, score in fused[:k]:
            c = self.chunks[idx]
            out.append({"chunk": c, "score": round(score, 5), "bm25": round(float(bm[idx]), 3),
                        "denso": round(float(dense[idx]), 4) if dense is not None else None})
        return out
