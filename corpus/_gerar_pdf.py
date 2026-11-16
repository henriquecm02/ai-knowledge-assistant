"""Converte a política de reembolso (Markdown) em PDF de 2 páginas, para exercitar a leitura de PDF."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

src = Path(__file__).parent / "_fontes" / "politica_reembolso.md"
out = Path(__file__).with_name("politica_reembolso.pdf")
lines = src.read_text(encoding="utf-8").splitlines()
c = canvas.Canvas(str(out), pagesize=A4)
y = A4[1] - 2 * cm
for i, line in enumerate(lines):
    if line.startswith("## 4."):  # força a quebra de página antes da seção 4
        c.showPage(); y = A4[1] - 2 * cm
    text = line.lstrip("#").strip()
    c.setFont("Helvetica-Bold" if line.startswith("#") else "Helvetica", 11 if line.startswith("#") else 10)
    for k in range(0, max(len(text), 1), 95):
        c.drawString(2 * cm, y, text[k:k + 95]); y -= 0.55 * cm
c.save()
print(out)
