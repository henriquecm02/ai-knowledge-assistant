"""Assistente: recupera trechos, (opcionalmente) reordena com o LLM e gera a resposta citando fontes.

Garantias em código, independentes do modelo:
- citações só podem apontar para trechos efetivamente enviados (as inválidas são descartadas);
- resposta que diz ter encontrado algo mas não cita nenhum trecho válido é marcada como sem fonte;
- sem nenhum trecho recuperado, o LLM nem é chamado: a resposta é "não encontrei".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLM, Usage
from .retrieval import Index

PROMPTS = Path(__file__).parent / "prompts"

ANSWER_SCHEMA = {"type": "object", "properties": {
    "resposta": {"type": "string"}, "encontrou": {"type": "boolean"},
    "citacoes": {"type": "array", "items": {"type": "integer"}}},
    "required": ["resposta", "encontrou", "citacoes"]}

RERANK_SCHEMA = {"type": "object", "properties": {"notas": {"type": "array", "items": {
    "type": "object", "properties": {"trecho": {"type": "integer"}, "nota": {"type": "integer", "minimum": 0,
                                                                             "maximum": 3}},
    "required": ["trecho", "nota"]}}}, "required": ["notas"]}

NOT_FOUND = "Não encontrei essa informação nos documentos disponíveis."


def prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8").strip()


def _passages(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i}] ({h['chunk'].citation})\n{h['chunk'].text}" for i, h in enumerate(hits, 1))


@dataclass
class Answer:
    pergunta: str
    resposta: str
    encontrou: bool
    fontes: list[dict]
    trechos_recuperados: list[dict]
    avisos: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def as_dict(self) -> dict:
        return {"pergunta": self.pergunta, "resposta": self.resposta, "encontrou": self.encontrou,
                "fontes": self.fontes, "avisos": self.avisos, "chamadas_llm": self.usage.calls,
                "trechos_recuperados": self.trechos_recuperados}


class Assistant:
    def __init__(self, index: Index, llm: LLM | None, k: int = 5, mode: str = "hibrido", rerank: bool = False,
                 min_rerank: int = 2):
        self.index, self.llm, self.k, self.mode, self.rerank, self.min_rerank = index, llm, k, mode, rerank, min_rerank

    def retrieve(self, question: str) -> list[dict]:
        mode = self.mode if self.index.vectors is not None else "bm25"
        return self.index.search(question, k=self.k * (2 if self.rerank else 1), mode=mode)

    def _rerank(self, question: str, hits: list[dict], usage: Usage) -> list[dict]:
        data, u = self.llm.json(prompt("rerank_v1"), f"Pergunta: {question}\n\n{_passages(hits)}", RERANK_SCHEMA)
        usage.add(u)
        notes = {n["trecho"]: n["nota"] for n in data.get("notas", [])}
        for i, h in enumerate(hits, 1):
            h["rerank"] = notes.get(i, 0)
        kept = [h for h in sorted(hits, key=lambda h: -h["rerank"]) if h["rerank"] >= self.min_rerank]
        return kept[: self.k]

    def ask(self, question: str) -> Answer:
        usage = Usage()
        hits = self.retrieve(question)
        if self.rerank and self.llm and hits:
            hits = self._rerank(question, hits, usage)
        summary = [{"citacao": h["chunk"].citation, "score": h["score"], "bm25": h["bm25"], "denso": h["denso"],
                    "rerank": h.get("rerank")} for h in hits]
        if not hits:
            return Answer(question, NOT_FOUND, False, [], summary, ["nenhum trecho recuperado"], usage)
        if self.llm is None:  # modo busca: sem chave, devolve os trechos mais relevantes
            return Answer(question, "Modo busca (sem LLM): veja os trechos mais relevantes abaixo.", True,
                          [{"n": i, "citacao": h["chunk"].citation, "texto": h["chunk"].text}
                           for i, h in enumerate(hits, 1)], summary, ["resposta não gerada: sem LLM"], usage)

        data, u = self.llm.json(prompt("answer_v1"), f"Pergunta: {question}\n\nTrechos:\n{_passages(hits)}",
                                ANSWER_SCHEMA)
        usage.add(u)
        valid = [c for c in dict.fromkeys(data.get("citacoes", [])) if 1 <= c <= len(hits)]
        warnings = []
        if len(valid) < len(data.get("citacoes", [])):
            warnings.append("citações para trechos inexistentes foram removidas")
        found = bool(data.get("encontrou"))
        if found and not valid:
            warnings.append("resposta sem trecho citado: trate como não verificada")
        sources = [{"n": c, "citacao": hits[c - 1]["chunk"].citation, "texto": hits[c - 1]["chunk"].text}
                   for c in valid]
        return Answer(question, data.get("resposta", "").strip() or NOT_FOUND, found and bool(valid), sources,
                      summary, warnings, usage)

    # ----------------------------------------------------------------- recursos extras

    def summarize_document(self, doc: str) -> str:
        parts = [c for c in self.index.chunks if c.doc == doc]
        if not parts or self.llm is None:
            return "Resumo indisponível (documento não encontrado ou sem LLM)."
        text = "\n\n".join(f"## {c.section}\n{c.text}" for c in parts)
        data, _ = self.llm.json("Resuma o documento em até 5 bullets com as regras mais importantes, copiando "
                                "valores e prazos exatamente. Não acrescente nada que não esteja no texto.",
                                text[:30000], {"type": "object", "properties": {"bullets": {"type": "array",
                                               "items": {"type": "string"}}}, "required": ["bullets"]})
        return "\n".join(f"- {b}" for b in data.get("bullets", []))
