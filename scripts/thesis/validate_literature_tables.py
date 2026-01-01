#!/usr/bin/env python3
"""
Fail-fast validation for thesis literature-review tables.

Checks:
- Every \\cite{...} key used in selected tables exists in `thesis/references.bib`.
- For tables that explicitly include a Year column, the TeX year matches the bib `year`.
- Warns (non-fatal) if a table row likely needs a cite but has none.

Usage:
  python3 scripts/thesis/validate_literature_tables.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BibEntry:
    key: str
    year: str | None
    title: str | None


def _parse_bib_entries(bib_text: str) -> dict[str, BibEntry]:
    entries: dict[str, BibEntry] = {}

    for match in re.finditer(r"(?m)^@\w+\s*\{", bib_text):
        start = match.start()
        open_brace = bib_text.find("{", start)
        if open_brace == -1:
            continue

        level = 0
        end = None
        for idx in range(open_brace, len(bib_text)):
            ch = bib_text[idx]
            if ch == "{":
                level += 1
            elif ch == "}":
                level -= 1
                if level == 0:
                    end = idx + 1
                    break

        if end is None:
            continue

        entry_text = bib_text[start:end]
        header_end = bib_text.find(",", open_brace)
        if header_end == -1 or header_end > end:
            continue

        key = bib_text[open_brace + 1 : header_end].strip()
        if not key:
            continue

        year_match = re.search(r"(?mi)^\s*year\s*=\s*\{([^}]*)\}", entry_text)
        title_match = re.search(r"(?mi)^\s*title\s*=\s*\{([^}]*)\}", entry_text)
        entries[key] = BibEntry(
            key=key,
            year=year_match.group(1).strip() if year_match else None,
            title=title_match.group(1).strip() if title_match else None,
        )

    return entries


def _extract_table(tex_text: str, label: str) -> str:
    label_token = f"\\label{{{label}}}"
    label_pos = tex_text.find(label_token)
    if label_pos == -1:
        raise ValueError(f"Missing table label in TeX: {label}")

    begin_pos = tex_text.rfind("\\begin{table", 0, label_pos)
    if begin_pos == -1:
        raise ValueError(f"Cannot find \\begin{{table}} before {label}")

    end_pos = tex_text.find("\\end{table}", label_pos)
    if end_pos == -1:
        raise ValueError(f"Cannot find \\end{{table}} after {label}")

    return tex_text[begin_pos : end_pos + len("\\end{table}")]


def _cite_keys_in_text(text: str) -> list[str]:
    keys: list[str] = []
    for block in re.findall(r"\\cite\{([^}]*)\}", text):
        for key in block.split(","):
            cleaned = key.strip()
            if cleaned:
                keys.append(cleaned)
    return keys


def _warn_rows_without_cites(table_text: str, label: str) -> list[str]:
    warnings: list[str] = []
    for line in table_text.splitlines():
        stripped = line.strip()
        if not stripped.endswith("\\\\"):
            continue
        if "\\textbf{" in stripped:
            continue
        if any(
            token in stripped
            for token in (
                "\\toprule",
                "\\midrule",
                "\\bottomrule",
                "\\multicolumn",
            )
        ):
            continue
        if "\\cite{" in stripped:
            continue
        if "This Work" in stripped:
            continue
        if stripped.startswith("Time-Windowing & Multiple"):
            continue
        if "&" not in stripped:
            continue
        warnings.append(f"{label}: row may be missing a citation: {stripped}")
    return warnings


def _year_checks(table_text: str, label: str, bib: dict[str, BibEntry]) -> list[str]:
    errors: list[str] = []
    for m in re.finditer(r"\\cite\{([^}]+)\}\s*&\s*(\d{4})\s*&", table_text):
        key_block, year_in_tex = m.group(1), m.group(2)
        key = key_block.split(",")[0].strip()
        entry = bib.get(key)
        if not entry:
            continue
        if entry.year and entry.year != year_in_tex:
            errors.append(
                f"{label}: year mismatch for {key}: TeX={year_in_tex}, bib={entry.year}"
            )
    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    tex_path = repo_root / "thesis" / "main.tex"
    bib_path = repo_root / "thesis" / "references.bib"

    tex_text = tex_path.read_text(encoding="utf-8", errors="ignore")
    bib_text = bib_path.read_text(encoding="utf-8", errors="ignore")

    bib = _parse_bib_entries(bib_text)
    if not bib:
        print("ERROR: failed to parse any bib entries", file=sys.stderr)
        return 2

    table_labels = [
        "tab:reconstruction_methods",
        "tab:pinns_foundations",
        "tab:pinns_turbulence",
        "tab:pinns_optimization",
    ]
    year_tables = {"tab:pinns_foundations", "tab:pinns_turbulence"}

    errors: list[str] = []
    warnings: list[str] = []

    for label in table_labels:
        table_text = _extract_table(tex_text, label)
        cite_keys = _cite_keys_in_text(table_text)
        if not cite_keys:
            errors.append(f"{label}: no citations found in table")
            continue

        for key in cite_keys:
            if key not in bib:
                errors.append(f"{label}: missing bib key: {key}")

        warnings.extend(_warn_rows_without_cites(table_text, label))
        if label in year_tables:
            errors.extend(_year_checks(table_text, label, bib))

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("ERRORS:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("OK: literature-review tables are consistent with thesis/references.bib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
