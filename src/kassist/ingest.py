"""Leitura de documentos e divisão em trechos (chunks) com metadados para citação.

Estratégia de chunking: respeita a estrutura do documento (seções Markdown, páginas de PDF,
parágrafos) e só corta dentro de uma seção quando ela passa do tamanho máximo, com sobreposição.
Cada trecho guarda documento, página e seção, o que permite citar "documento · seção · página".
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

SUPPORTED = {".md", ".txt", ".pdf", ".docx"}


@dataclass
class Chunk:
    id: str
    doc: str
    section: str
    page: int | None
    text: str

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def citation(self) -> str:
        p = f" · p. {self.page}" if self.page else ""
        return f"{self.doc} · {self.section}{p}"


def _pages(path: Path) -> list[tuple[int | None, str]]:
    suf = path.suffix.lower()
    if suf == ".pdf":
        from pypdf import PdfReader
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(str(path)).pages)]
    if suf == ".docx":
        from docx import Document
        d = Document(str(path))
        lines = []
        for p in d.paragraphs:
            if p.text.strip():
                level = p.style.name.replace("Heading ", "") if p.style.name.startswith("Heading") else None
                lines.append(("#" * int(level) + " " if level and level.isdigit() else "") + p.text)
        return [(None, "\n".join(lines))]
    return [(None, path.read_text(encoding="utf-8", errors="replace"))]


HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
PDF_HEADING = re.compile(r"^(\d+(?:\.\d+)*\.?\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ].{2,80})$")


def _sections(text: str, is_pdf: bool) -> list[tuple[str, str]]:
    """Divide em (título da seção, corpo). PDFs sem marcação usam linhas numeradas como títulos."""
    out: list[tuple[str, list[str]]] = [("Introdução", [])]
    for line in text.splitlines():
        m = HEADING.match(line.strip())
        if m and len(m.group(1)) >= 2:
            out.append((m.group(2).strip(), []))
        elif is_pdf and PDF_HEADING.match(line.strip()):
            out.append((line.strip(), []))
        elif m:  # título do documento (#): vira contexto da introdução
            out[0][1].append(m.group(2).strip())
        else:
            out[-1][1].append(line)
    return [(t, "\n".join(b).strip()) for t, b in out if "\n".join(b).strip()]


def _split(body: str, max_chars: int, overlap: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=- )", body) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 1 > max_chars:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n" + p if overlap else p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    out = []
    for c in chunks:  # parágrafo único gigante: corte duro com sobreposição
        while len(c) > max_chars * 1.5:
            out.append(c[:max_chars])
            c = c[max_chars - overlap:]
        out.append(c)
    return out


def chunk_document(path: Path, max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    chunks = []
    title_prefix = ""
    for page, text in _pages(path):
        for section, body in _sections(text, path.suffix.lower() == ".pdf"):
            if section == "Introdução" and not title_prefix:
                title_prefix = body.splitlines()[0][:120] if body else ""
            for part in _split(body, max_chars, overlap):
                cid = hashlib.sha1(f"{path.name}|{page}|{section}|{part[:80]}".encode()).hexdigest()[:12]
                chunks.append(Chunk(cid, path.name, section, page, part))
    return chunks


def load_corpus(folder: Path, **kw) -> list[Chunk]:
    out = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in SUPPORTED and p.is_file():
            out += chunk_document(p, **kw)
    return out
