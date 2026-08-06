"""Generate the README diagram: many Fabric sources, one task, one workbook.

Two SVGs are emitted, light and dark, so the README can use <picture> and read
correctly in either GitHub theme. They come from one definition here so they
cannot drift apart.

    python docs/assets/make_multisource_diagram.py
"""

from __future__ import annotations

import pathlib

W, H = 1060, 610
FONT = ("ui-sans-serif, -apple-system, 'Segoe UI', Roboto, Helvetica, "
        "Arial, sans-serif")
MONO = "ui-monospace, SFMono-Regular, 'Cascadia Mono', Menlo, Consolas, monospace"

THEMES = {
    "light": dict(
        text="#1f2328", muted="#59636e", panel="#ffffff", panelEdge="#d1d9e0",
        band="#f6f8fa", accent="#0969da", accentSoft="#ddf4ff",
        good="#1a7f37", goodSoft="#dafbe1", arrow="#8c959f", code="#1f2328",
    ),
    "dark": dict(
        text="#e6edf3", muted="#9198a1", panel="#0d1117", panelEdge="#3d444d",
        band="#151b23", accent="#4493f8", accentSoft="#121d2f",
        good="#3fb950", goodSoft="#0f2913", arrow="#6e7681", code="#e6edf3",
    ),
}

SOURCES = [
    ("2 semantic models", "Manufacturing Ops, ARR", "accent"),
    ("5 CSV", "targets, headcount, FX rates", "plain"),
    ("3 PDF", "board memo, audit, contract", "plain"),
    ("2 XLSX", "budget, prior quarter", "plain"),
]

# Two-space indent and a 10.5px face: at four spaces the SemanticModel line is
# wider than the box it sits in.
CODE = [
    "RLM.task(",
    "  task=BRIEF,",
    "  inputs={",
    '    "mfg":  SemanticModel("Manufacturing Ops"),',
    '    "arr":  SemanticModel("ARR Model SF"),',
    '    "csvs": [File(p) for p in csv_paths],',
    '    "pdfs": [File(p) for p in pdf_paths],',
    '    "xlsx": [File(p) for p in xlsx_paths],',
    "  },",
    "  output_validator=workbook_exists,",
    ").run()",
]

# Keep these under ~26 characters: the output panel fits about that at 11px.
OUTPUT = [
    "Summary, By Plant, Notes",
    "currency, percent formats",
    "frozen, sized headers",
    "conditional formatting",
]


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build(theme: str) -> str:
    c = THEMES[theme]
    o: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" role="img" '
        f'aria-label="Several Fabric Lakehouse sources feeding one fabric-rlm '
        f'task that writes a formatted Excel workbook back to the Lakehouse">',
        '<defs>',
        f'<marker id="a{theme}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{c["arrow"]}"/></marker>',
        '</defs>',
    ]

    def text(x, y, s, size=14, fill=None, weight="400", family=FONT, anchor="start"):
        o.append(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill or c["text"]}" '
            f'text-anchor="{anchor}">{esc(s)}</text>')

    def box(x, y, w, h, fill=None, stroke=None, r=10, sw=1):
        o.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
            f'fill="{fill or c["panel"]}" stroke="{stroke or c["panelEdge"]}" '
            f'stroke-width="{sw}"/>')

    # ---- left: the lakehouse -------------------------------------------
    box(24, 46, 300, 500, fill=c["band"])
    text(44, 78, "Microsoft Fabric Lakehouse", 15, weight="600")
    text(44, 100, "one workspace, one attached lakehouse", 12, c["muted"])

    y = 122
    for label, detail, kind in SOURCES:
        fill = c["accentSoft"] if kind == "accent" else c["panel"]
        edge = c["accent"] if kind == "accent" else c["panelEdge"]
        box(44, y, 260, 88, fill=fill, stroke=edge)
        text(64, y + 34, label, 15, weight="600")
        text(64, y + 58, detail, 12, c["muted"])
        y += 100

    # ---- arrows in ------------------------------------------------------
    # Start outside the lakehouse panel (right edge 324), converge on the
    # vertical centre of the task panel (118 + 356/2).
    for i in range(4):
        cy = 122 + i * 100 + 44
        o.append(
            f'<path d="M332 {cy} C 366 {cy}, 366 296, 392 296" fill="none" '
            f'stroke="{c["arrow"]}" stroke-width="1.6" '
            f'marker-end="url(#a{theme})" opacity="0.75"/>')

    # ---- middle: the task ----------------------------------------------
    box(400, 118, 372, 356)
    text(422, 150, "fabric-rlm", 15, weight="600")
    text(422, 172, "one task, all sources bound at once", 12, c["muted"])

    box(418, 188, 336, 268, fill=c["band"], r=8)
    ty = 212
    for line in CODE:
        text(434, ty, line, 10.5, c["code"], family=MONO)
        ty += 22

    text(422, 500, "Writes Python in a sandbox. Queries each model", 12, c["muted"])
    text(422, 520, "with DAX, reads the files, builds the workbook.", 12, c["muted"])

    # ---- arrow out ------------------------------------------------------
    o.append(
        f'<path d="M772 296 L 828 296" fill="none" stroke="{c["arrow"]}" '
        f'stroke-width="1.8" marker-end="url(#a{theme})"/>')

    # ---- right: the output ---------------------------------------------
    box(836, 168, 214, 256, fill=c["goodSoft"], stroke=c["good"])
    text(856, 202, "ops_review.xlsx", 14, weight="600", family=MONO)
    text(856, 224, "written back to the", 12, c["muted"])
    text(856, 242, "Lakehouse", 12, c["muted"])

    oy = 278
    for line in OUTPUT:
        o.append(f'<circle cx="861" cy="{oy - 4}" r="2.5" fill="{c["good"]}"/>')
        text(872, oy, line, 11)
        oy += 26

    text(856, 402, "Files/reports/", 11, c["muted"], family=MONO)

    # ---- footer ---------------------------------------------------------
    box(24, 562, 1012, 34, fill=c["band"], r=8)
    text(530, 584,
         "The model decides which source answers which part of the brief. "
         "Nothing is pre-joined.",
         12.5, c["muted"], anchor="middle")

    o.append("</svg>")
    return "\n".join(o)


def main() -> None:
    here = pathlib.Path(__file__).parent
    for theme in THEMES:
        path = here / f"multisource-{theme}.svg"
        path.write_text(build(theme), encoding="utf-8", newline="\n")
        print(f"wrote {path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
