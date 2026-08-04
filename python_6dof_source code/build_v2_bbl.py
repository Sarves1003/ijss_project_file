#!/usr/bin/env python3
"""
Hand-generates main_v2.bbl in IEEEtran citation style from the 80 references
actually cited in the restructured manuscript, because this sandbox's TeX
Live install lacks IEEEtran.bst and has no reliable network/root access to
fetch it (see algorithm.sty/algorithmic.sty shims for the same constraint).
Trims references_v2.bib down to exactly the cited set and writes both the
trimmed .bib (for the record) and a formatted main_v2.bbl.
"""
import os
import re

LATEX = os.path.join(os.path.dirname(__file__), "..", "latex")
BIB_IN = os.path.join(LATEX, "references_v2.bib")
BIB_OUT = os.path.join(LATEX, "references_v2_cited.bib")
BBL_OUT = os.path.join(LATEX, "main_v2.bbl")
CITED_KEYS_FILE = "/tmp/final_cited.txt"


def parse_bib(path):
    text = open(path, encoding="utf-8").read()
    entries = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        etype, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*(?=\n\s*\w+\s*=|\s*$)", body, re.S):
            fields[fm.group(1).lower()] = re.sub(r"\s+", " ", fm.group(2)).strip()
        entries[key] = {"type": etype, **fields}
    return entries


def strip_latex_accents(s):
    # Collapse LaTeX accent macros like {\'o}, {\~n}, {\^e} etc. to the plain letter
    s = re.sub(r"\{\\[\'\"\^~`.=uvHc]\s*([a-zA-Z])\}", r"\1", s)
    s = re.sub(r"\\[\'\"\^~`.=uvHc]\s*\{?([a-zA-Z])\}?", r"\1", s)
    # No-argument special-letter macros, e.g. {\o} -> o, {\O} -> O, \ss -> ss
    special = {"o": "o", "O": "O", "l": "l", "L": "L", "aa": "a", "AA": "A", "ss": "ss", "ae": "ae", "AE": "AE"}
    for macro, plain in special.items():
        s = re.sub(r"\{\\" + macro + r"\}", plain, s)
        s = re.sub(r"\\" + macro + r"(?![a-zA-Z])", plain, s)
    s = s.replace("{", "").replace("}", "")
    return s


def format_author(name):
    name = strip_latex_accents(name.strip())
    parts = name.split()
    if len(parts) == 1:
        return parts[0]
    last = parts[-1]
    initials = []
    for p in parts[:-1]:
        p_clean = p.replace(".", "")
        if not p_clean:
            continue
        initials.append(p_clean[0] + ".")
    return "~".join(["".join(initials)]) + f"~{last}" if initials else last


def format_authors(author_field):
    if not author_field:
        return "Anonymous"
    authors = [a.strip() for a in re.split(r"\s+and\s+", author_field) if a.strip()]
    formatted = [format_author(a) for a in authors]
    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    else:
        return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"


def format_entry(key, e):
    authors = format_authors(e.get("author", ""))
    title = e.get("title", "").rstrip(".")
    year = e.get("year", "n.d.")
    journal = e.get("journal") or e.get("booktitle") or e.get("publisher") or ""
    vol = e.get("volume", "")
    num = e.get("number", "")
    pages = e.get("pages", "")

    parts = [f"{authors}, ``{title},''"]
    if journal:
        parts.append(f"\\emph{{{journal}}},")
    piece = []
    if vol:
        piece.append(f"vol.~{vol}")
    if num:
        piece.append(f"no.~{num}")
    if piece:
        parts.append(", ".join(piece) + ",")
    if pages:
        parts.append(f"pp.\\ {pages},")
    parts.append(f"{year}.")

    line = " ".join(parts)
    line = line.replace(",,", ",").replace(", .", ".")
    return line


def main():
    all_entries = parse_bib(BIB_IN)
    cited = [k.strip() for k in open(CITED_KEYS_FILE).read().split() if k.strip()]

    missing = [k for k in cited if k not in all_entries]
    if missing:
        raise SystemExit(f"Missing bib entries for cited keys: {missing}")

    cited_sorted = sorted(cited, key=lambda k: (-int(all_entries[k].get("year", 0) or 0), k))

    with open(BIB_OUT, "w", encoding="utf-8") as f:
        for k in cited_sorted:
            e = all_entries[k]
            f.write(f"@{e['type']}{{{k},\n")
            for field, val in e.items():
                if field == "type":
                    continue
                f.write(f"  {field} = {{{val}}},\n")
            f.write("}\n\n")

    bbl_lines = [
        "% Hand-generated in IEEEtran citation style (see build_v2_bbl.py):",
        "% this build environment has no IEEEtran.bst and no reliable network/root",
        "% access to install it. Every entry traces to references_v2.bib (real DOIs).",
        "\\begin{thebibliography}{80}",
        "\\footnotesize",
        "\\setlength{\\itemsep}{0pt}",
        "\\setlength{\\parskip}{0pt}",
        "\\setlength{\\parsep}{0pt}",
        "\\renewcommand{\\baselinestretch}{0.96}\\selectfont",
        "\\providecommand{\\url}[1]{#1}",
        "\\csname url@samestyle\\endcsname",
        "\\providecommand{\\newblock}{\\relax}",
        "",
    ]
    for k in cited_sorted:
        e = all_entries[k]
        bbl_lines.append(f"\\bibitem{{{k}}}")
        bbl_lines.append(format_entry(k, e))
        bbl_lines.append("")
    bbl_lines.append("\\normalsize")
    bbl_lines.append("\\end{thebibliography}")

    with open(BBL_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(bbl_lines))

    print(f"Wrote {len(cited_sorted)} entries to {BIB_OUT} and {BBL_OUT}")


if __name__ == "__main__":
    main()
