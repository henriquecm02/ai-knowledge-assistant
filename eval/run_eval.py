"""Avaliação reproduzível da recuperação e (com chave) das respostas.

  python eval/run_eval.py                 # BM25: roda sem chave
  python eval/run_eval.py --llm           # + denso, híbrido, reranking e respostas (exige GEMINI_API_KEY)

Métricas
  Recuperação: Recall@1/3/5 e MRR do trecho correto (documento + seção), por tipo de pergunta.
  Respostas:   contém o fato esperado · cita o trecho correto · abstém-se quando não há resposta.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from kassist.assistant import Assistant
from kassist.ingest import load_corpus
from kassist.retrieval import Index
from kassist.retrieval import fold

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = json.loads((ROOT / "eval" / "perguntas.json").read_text(encoding="utf-8"))


def is_gold(q: dict, citation_doc: str, citation_section: str) -> bool:
    return citation_doc == q["doc"] and q["secao_contem"].lower() in citation_section.lower()


def eval_retrieval(index: Index, mode: str) -> dict:
    out = {}
    for tipo in ("direta", "parafrase"):
        sub = [q for q in QUESTIONS if q["tipo"] == tipo]
        hits = {1: 0, 3: 0, 5: 0}
        mrr, misses = 0.0, []
        for q in sub:
            res = index.search(q["pergunta"], k=5, mode=mode)
            ranks = [i for i, r in enumerate(res) if is_gold(q, r["chunk"].doc, r["chunk"].section)]
            if ranks:
                for k in hits:
                    hits[k] += ranks[0] < k
                mrr += 1 / (ranks[0] + 1)
            if not ranks or ranks[0] > 0:
                misses.append({"id": q["id"], "pergunta": q["pergunta"],
                               "top1": res[0]["chunk"].citation if res else None})
        out[tipo] = {"n": len(sub), **{f"recall@{k}": round(v / len(sub), 3) for k, v in hits.items()},
                     "mrr": round(mrr / len(sub), 3), "erros_top1": misses}
    return out


def eval_answers(assistant: Assistant) -> dict:
    rows = []
    for q in QUESTIONS:
        t0 = time.perf_counter()
        a = assistant.ask(q["pergunta"])
        secs = time.perf_counter() - t0
        if q["tipo"] == "sem_resposta":
            ok = not a.encontrou
            row = {"id": q["id"], "tipo": q["tipo"], "absteve": not a.encontrou, "correta": ok}
        else:
            fact = fold(q["resposta_contem"]) in fold(a.resposta)
            cites_gold = any(s["citacao"].startswith(q["doc"]) and q["secao_contem"].lower() in s["citacao"].lower()
                             for s in a.fontes)
            row = {"id": q["id"], "tipo": q["tipo"], "fato_correto": fact, "cita_trecho_correto": cites_gold,
                   "correta": fact and cites_gold}
        rows.append({**row, "segundos": round(secs, 2), "chamadas_llm": a.usage.calls, "resposta": a.resposta,
                     "avisos": a.avisos})
    agg = {}
    for tipo in ("direta", "parafrase", "sem_resposta"):
        sub = [r for r in rows if r["tipo"] == tipo]
        agg[tipo] = {"n": len(sub), "acerto": round(sum(r["correta"] for r in sub) / len(sub), 3)}
    return {"por_tipo": agg, "linhas": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()
    chunks = load_corpus(ROOT / "corpus")
    report: dict = {"data": datetime.now().isoformat(timespec="seconds"), "trechos": len(chunks),
                    "perguntas": len(QUESTIONS), "recuperacao": {}}
    report["recuperacao"]["bm25"] = eval_retrieval(Index(chunks), "bm25")

    if args.llm:
        load_dotenv(ROOT / ".env")
        load_dotenv(Path.home() / ".config" / "portfolio" / ".env")
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("--llm exige GEMINI_API_KEY")
        from kassist.llm import GeminiLLM
        from kassist.retrieval import GeminiEmbedder
        index = Index(chunks, GeminiEmbedder(key))
        report["embedder"] = index.embedder.name
        for mode in ("denso", "hibrido"):
            report["recuperacao"][mode] = eval_retrieval(index, mode)
        llm = GeminiLLM(key, os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
        report["modelo"] = llm.model
        report["respostas"] = {"hibrido": eval_answers(Assistant(index, llm)),
                               "hibrido+rerank": eval_answers(Assistant(index, llm, rerank=True))}

    out = ROOT / "eval" / "resultados"
    out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    (out / f"{stamp}{'_llm' if args.llm else '_bm25'}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"| Modo | Tipo | R@1 | R@3 | R@5 | MRR |\n|---|---|---|---|---|---|")
    for mode, r in report["recuperacao"].items():
        for tipo, m in r.items():
            print(f"| {mode} | {tipo} ({m['n']}) | {m['recall@1']:.2f} | {m['recall@3']:.2f} | {m['recall@5']:.2f} "
                  f"| {m['mrr']:.2f} |")
    for name, res in report.get("respostas", {}).items():
        print(name, {k: v["acerto"] for k, v in res["por_tipo"].items()})


if __name__ == "__main__":
    main()
