#!/usr/bin/env python3
"""
Merges the 50-entry curated "core" bibliography (foundational math/control/
vision references, mostly pre-2022, needed for the derivations) with the
150-entry OpenAlex-sourced literature-survey bibliography (real papers with
DOIs, heavier in 2018-2025), deduplicates by normalized title, adds a small
set of additional 2024-2025 papers verified individually against the
Crossref API, and writes the final references.bib for the restructured
manuscript. No fabricated entries -- every record here traces to a real DOI.
"""
import os
import re

BASE = os.path.join(os.path.dirname(__file__), "..")
CORE_BIB = os.path.join(BASE, "latex", "references_50.bib")
SURVEY_BIB = os.path.join(BASE, "bibliography", "references.bib")
OUT_BIB = os.path.join(BASE, "latex", "references_v2.bib")

NEW_ENTRIES = r"""
@article{ElgueaAguinaco2024,
  author = {{\'I}{\~n}igo Elguea-Aguinaco and Ibai Inziarte-Hidalgo and Simon B{\o}gh and Nestor Arana-Arexolaleiba},
  title = {A Review on Reinforcement Learning for Motion Planning of Robotic Manipulators},
  journal = {International Journal of Intelligent Systems},
  volume = {2024},
  pages = {1636497},
  year = {2024},
  doi = {10.1155/int/1636497}
}

@article{Santos2024VisionPickPlace,
  author = {Adriano A. Santos and Cas Schreurs and Ant{\'o}nio Ferreira da Silva and Filipe Pereira and Carlos Felgueiras and Ant{\'o}nio M. Lopes and Jos{\'e} Machado},
  title = {Integration of Artificial Vision and Image Processing into a Pick and Place Collaborative Robotic System},
  journal = {Journal of Intelligent \& Robotic Systems},
  volume = {110},
  number = {4},
  pages = {159},
  year = {2024},
  doi = {10.1007/s10846-024-02195-z}
}

@article{Vrabel2025HybridPSOGWO,
  author = {Robert Vrabel},
  title = {Hybrid Particle Swarm and Grey Wolf Optimization for Robust Feedback Control of Nonlinear Systems},
  journal = {Automation},
  volume = {6},
  number = {4},
  pages = {89},
  year = {2025},
  doi = {10.3390/automation6040089}
}

@article{Oglah2025FOPIDGWO,
  author = {Ahmed A. Oglah and Mohammed S. Saleh and Abidaoun H. Shallal},
  title = {Design of Fractional--PID Controller Based on Grey Wolf Optimization for Robotic Manipulator},
  journal = {Tikrit Journal of Engineering Sciences},
  volume = {32},
  number = {3},
  pages = {1--8},
  year = {2025},
  doi = {10.25130/tjes.32.3.9}
}
"""


def normalize_title(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_bib(path):
    text = open(path, encoding="utf-8").read()
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        etype, key, body = m.group(1), m.group(2).strip(), m.group(3)
        title_m = re.search(r"title\s*=\s*\{(.*?)\}\s*,?\s*\n", body, re.S)
        year_m = re.search(r"year\s*=\s*\{?(\d{4})", body)
        title = title_m.group(1).strip() if title_m else ""
        year = int(year_m.group(1)) if year_m else None
        entries.append({"type": etype, "key": key, "body": body, "title": title,
                         "year": year, "raw": m.group(0)})
    return entries


core = parse_bib(CORE_BIB)
survey = parse_bib(SURVEY_BIB)
new = parse_bib("/dev/stdin") if False else []

# parse NEW_ENTRIES from the in-memory string using the same regex
new = []
for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", NEW_ENTRIES, re.S):
    etype, key, body = m.group(1), m.group(2).strip(), m.group(3)
    title_m = re.search(r"title\s*=\s*\{(.*?)\}\s*,?\s*\n", body, re.S)
    year_m = re.search(r"year\s*=\s*\{?(\d{4})", body)
    new.append({"type": etype, "key": key, "body": body,
                "title": title_m.group(1).strip() if title_m else "",
                "year": int(year_m.group(1)) if year_m else None,
                "raw": m.group(0)})

all_entries = core + survey + new
seen_titles = set()
seen_keys = set()
final = []
for e in all_entries:
    nt = normalize_title(e["title"])
    if not nt or nt in seen_titles:
        continue
    key = e["key"]
    suffix = 1
    orig_key = key
    while key in seen_keys:
        suffix += 1
        key = f"{orig_key}{suffix}"
    if key != orig_key:
        e["raw"] = e["raw"].replace(f"{{{orig_key},", f"{{{key},", 1)
    seen_titles.add(nt)
    seen_keys.add(key)
    final.append(e)

years = [e["year"] for e in final if e["year"]]
recent = [e for e in final if e["year"] and e["year"] >= 2022]

with open(OUT_BIB, "w", encoding="utf-8") as f:
    f.write("% Final curated bibliography for the restructured IJSS manuscript.\n")
    f.write(f"% {len(final)} unique entries, {len(recent)} from 2022-2026.\n")
    f.write("% Merged from: latex/references_50.bib (foundational/core), \n")
    f.write("% bibliography/references.bib (150-entry OpenAlex literature survey),\n")
    f.write("% and 4 additional entries individually verified via the Crossref API.\n\n")
    for e in final:
        f.write(e["raw"].strip() + "\n\n")

print(f"Total unique entries: {len(final)}")
print(f"2022-2026 entries: {len(recent)}")
print(f"Written to {OUT_BIB}")
