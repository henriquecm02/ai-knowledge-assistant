"""Testes offline: ingestão, BM25, fusão híbrida e as garantias do assistente com LLM roteirizado."""
from __future__ import annotations

from pathlib import Path

import pytest

from kassist.assistant import NOT_FOUND, Assistant
from kassist.ingest import chunk_document, load_corpus
from kassist.llm import ScriptedLLM
from kassist.retrieval import BM25, HashingEmbedder, Index, rrf, tokenize

CORPUS = Path(__file__).resolve().parents[1] / "corpus"


@pytest.fixture(scope="module")
def chunks():
    return load_corpus(CORPUS)


def test_pdf_chunks_keep_page_and_section(chunks):
    pdf = [c for c in chunks if c.doc.endswith(".pdf")]
    by_section = {c.section: c.page for c in pdf}
    assert by_section["2. Limites"] == 1 and by_section["4. Aprovação e pagamento"] == 2
    assert "p. 2" in next(c for c in pdf if c.page == 2).citation


def test_long_section_is_split_with_overlap(tmp_path):
    body = "\n\n".join(f"Parágrafo {i} " + "palavra " * 40 for i in range(12))
    p = tmp_path / "longo.md"
    p.write_text(f"# Doc\n\n## Seção grande\n{body}", encoding="utf-8")
    parts = [c for c in chunk_document(p, max_chars=600, overlap=100) if c.section == "Seção grande"]
    assert len(parts) > 3 and all(len(c.text) <= 900 for c in parts)
    assert parts[1].text[:60] in parts[0].text  # sobreposição


def test_tokenizer_folds_accents_and_stems():
    assert tokenize("Licenças e Férias") == tokenize("licenca ferias")


def test_bm25_ranks_exact_term_first():
    bm = BM25(["vale refeição de R$ 45,00", "política de senhas", "férias em três períodos"])
    assert bm.scores("qual o valor do vale-refeição").argmax() == 0


def test_rrf_rewards_agreement():
    fused = dict(rrf([[1, 2, 3], [3, 1, 2]]))
    assert fused[1] > fused[2] and fused[3] > fused[2]


def test_hybrid_index_runs_with_any_embedder(chunks):
    idx = Index(chunks, HashingEmbedder())
    res = idx.search("limite de hospedagem em capitais", k=3, mode="hibrido")
    assert res[0]["chunk"].section == "2. Limites" and res[0]["denso"] is not None


def _assistant(chunks, response):
    llm = ScriptedLLM(json_fn=lambda prompt, schema: response)
    return Assistant(Index(chunks), llm, mode="bm25"), llm


def test_answer_keeps_only_valid_citations(chunks):
    a, _ = _assistant(chunks, {"resposta": "Até R$ 120,00 por dia.", "encontrou": True, "citacoes": [1, 99]})
    ans = a.ask("Qual o limite de alimentação por dia em viagem?")
    assert [s["n"] for s in ans.fontes] == [1]
    assert "citações para trechos inexistentes foram removidas" in ans.avisos
    assert ans.fontes[0]["citacao"].startswith("politica_reembolso.pdf · 2. Limites")


def test_found_without_citation_is_not_trusted(chunks):
    a, _ = _assistant(chunks, {"resposta": "Sim.", "encontrou": True, "citacoes": []})
    ans = a.ask("A VPN é obrigatória em Wi-Fi público?")
    assert ans.encontrou is False and any("sem trecho citado" in w for w in ans.avisos)


def test_no_retrieval_means_no_llm_call(chunks):
    calls = []
    llm = ScriptedLLM(json_fn=lambda p, s: calls.append(1) or {})
    ans = Assistant(Index(chunks), llm, mode="bm25").ask("xyzzy quux")
    assert ans.resposta == NOT_FOUND and not calls


def test_search_mode_without_llm_returns_passages(chunks):
    ans = Assistant(Index(chunks), None, mode="bm25").ask("prazo para atestado médico")
    assert ans.fontes and "48 horas" in ans.fontes[0]["texto"]


def test_rerank_filters_low_scores(chunks):
    def fn(prompt, schema):
        if "notas" in schema["properties"]:
            return {"notas": [{"trecho": 1, "nota": 0}, {"trecho": 2, "nota": 3}]}
        return {"resposta": "ok", "encontrou": True, "citacoes": [1]}
    a = Assistant(Index(chunks), ScriptedLLM(json_fn=fn), k=3, mode="bm25", rerank=True)
    ans = a.ask("Qual o tamanho mínimo da senha?")
    assert len(ans.trechos_recuperados) == 1 and ans.trechos_recuperados[0]["rerank"] == 3
