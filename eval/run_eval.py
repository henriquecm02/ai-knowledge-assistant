"""Avaliação reproduzível da recuperação e (com chave) das respostas.

  python eval/run_eval.py                 # BM25: roda sem chave
  python eval/run_eval.py --llm           # + denso, híbrido, reranking e respostas (exige GEMINI_API_KEY)
  python eval/run_eval.py --embeddings    # só recuperação densa/híbrida (usa embeddings, sem geração)
  python eval/run_eval.py --llm --amostra # respostas só numa amostra fixa (poupa a cota gratuita)
  python eval/run_eval.py --llm --rerank  # 38 perguntas, com e sem reordenação; RETOMÁVEL (veja abaixo)
  python eval/run_eval.py --llm --denso   # + respostas com a busca densa (o cache pula o que já foi medido)

Retomável: cada resposta é guardada em eval/resultados/respostas_<modelo>.json. Ao esgotar a cota diária o script
salva o que fez e para; rodar de novo no dia seguinte continua de onde parou, sempre com o MESMO modelo.

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


# Amostra fixa para caber na cota gratuita: as 4 sem resposta + 6 paráfrases + 4 diretas (as primeiras de cada tipo).
SAMPLE = ([q for q in QUESTIONS if q["tipo"] == "sem_resposta"] + [q for q in QUESTIONS if q["tipo"] == "parafrase"][:6]
          + [q for q in QUESTIONS if q["tipo"] == "direta"][:4])


def eval_answers(assistant: Assistant, questions: list[dict] | None = None, cache: dict | None = None,
                 save=lambda: None) -> dict:
    """Responde as perguntas ainda sem resultado em `cache` (id -> linha). Para sem erro ao esgotar a cota."""
    from kassist.llm import QuotaExceeded
    cache = {} if cache is None else cache
    qs = questions or QUESTIONS
    for q in qs:
        if str(q["id"]) in cache:
            continue
        t0 = time.perf_counter()
        try:
            a = assistant.ask(q["pergunta"])
        except QuotaExceeded as e:
            print(f"  ⏸ cota esgotada em {q['id']}; {len(cache)}/{len(qs)} respondidas. Retome depois. ({str(e)[:80]})")
            break
        secs = time.perf_counter() - t0
        if q["tipo"] == "sem_resposta":
            row = {"id": q["id"], "tipo": q["tipo"], "absteve": not a.encontrou, "correta": not a.encontrou}
        else:
            fact = fold(q["resposta_contem"]) in fold(a.resposta)
            cites_gold = any(s["citacao"].startswith(q["doc"]) and q["secao_contem"].lower() in s["citacao"].lower()
                             for s in a.fontes)
            row = {"id": q["id"], "tipo": q["tipo"], "fato_correto": fact, "cita_trecho_correto": cites_gold,
                   "correta": fact and cites_gold}
        cache[str(q["id"])] = {**row, "segundos": round(secs, 2), "chamadas_llm": a.usage.calls, "resposta": a.resposta,
                               "avisos": a.avisos}
        save()
    rows = [cache[str(q["id"])] for q in qs if str(q["id"]) in cache]
    agg = {}
    for tipo in ("direta", "parafrase", "sem_resposta"):
        sub = [r for r in rows if r["tipo"] == tipo]
        total = len([q for q in qs if q["tipo"] == tipo])
        agg[tipo] = {"n": len(sub), "de": total,
                     "acerto": round(sum(r["correta"] for r in sub) / len(sub), 3) if sub else None}
    return {"por_tipo": agg, "linhas": rows, "completo": len(rows) == len(qs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--embeddings", action="store_true", help="só recuperação com embeddings, sem gerar respostas")
    ap.add_argument("--amostra", action="store_true", help="respostas apenas na amostra fixa de 14 perguntas")
    ap.add_argument("--rerank", action="store_true", help="também avalia híbrido+reordenação (dobra as chamadas)")
    ap.add_argument("--denso", action="store_true", help="também avalia as respostas com a busca densa (sem BM25)")
    args = ap.parse_args()
    chunks = load_corpus(ROOT / "corpus")
    report: dict = {"data": datetime.now().isoformat(timespec="seconds"), "trechos": len(chunks),
                    "perguntas": len(QUESTIONS), "recuperacao": {}}
    report["recuperacao"]["bm25"] = eval_retrieval(Index(chunks), "bm25")

    if args.llm or args.embeddings:
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
        if args.llm:
            llm = GeminiLLM(key, os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
            report["modelo"] = llm.model
            qs = SAMPLE if args.amostra else None
            cache_file = ROOT / "eval" / "resultados" / f"respostas_{llm.model}.json"
            store = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}

            def save():
                cache_file.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
            modos = [("hibrido", "hibrido", False)] + ([("hibrido+rerank", "hibrido", True)] if args.rerank else []) \
                + ([("denso", "denso", False)] if args.denso else [])
            report["respostas"] = {}
            for nome, modo, rr in modos:
                report["respostas"][nome] = eval_answers(Assistant(index, llm, mode=modo, rerank=rr), qs,
                                                         store.setdefault(nome, {}), save)
            report["amostra_respostas"] = len(qs) if qs else len(QUESTIONS)

    out = ROOT / "eval" / "resultados"
    out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    (out / f"{stamp}{'_llm' if args.llm else '_emb' if args.embeddings else '_bm25'}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"| Modo | Tipo | R@1 | R@3 | R@5 | MRR |\n|---|---|---|---|---|---|")
    for mode, r in report["recuperacao"].items():
        for tipo, m in r.items():
            print(f"| {mode} | {tipo} ({m['n']}) | {m['recall@1']:.2f} | {m['recall@3']:.2f} | {m['recall@5']:.2f} "
                  f"| {m['mrr']:.2f} |")
    for name, res in report.get("respostas", {}).items():
        print(name, {k: f"{v['acerto']} ({v['n']}/{v['de']})" for k, v in res["por_tipo"].items()})


if __name__ == "__main__":
    main()
