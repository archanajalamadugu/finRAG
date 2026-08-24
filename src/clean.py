"""
Ingestion + cleaning for SEC 10-K filings.

Design notes (these justify the "Ingestion + cleaning" framework field):
  * 10-K HTML is mostly layout markup. We strip script/style, XBRL hidden
    blocks, and page-break artifacts before extracting anything.
  * Filings use <table> for BOTH real financial data and page layout.
    We classify tables and keep only data tables, rendered as pipe-delimited
    text so numbers stay adjacent to their row and column labels.
  * Data tables are emitted as ATOMIC blocks. Nothing downstream is allowed
    to split a table across two chunks -- that is the single biggest cause
    of wrong numeric answers in financial RAG.
  * We tag every block with its "Item" section (Item 1A Risk Factors,
    Item 7 MD&A, Item 8 Financial Statements, ...) so citations can name a
    real location in the filing rather than an opaque chunk id.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from bs4 import BeautifulSoup

# ---------------------------------------------------------------- constants

DROP_TAGS = ("script", "style", "noscript", "head", "meta", "link")

# Canonical 10-K section headings we care about.
ITEM_RE = re.compile(
    r"^\s*item\s+(\d{1,2})\s*([A-C])?\s*[.\-:—]?\s*(.{0,80})$",
    re.IGNORECASE,
)

# Some document families number their sections as Rules rather than Items --
# "RULE 19: FLIGHT DELAYS". Supported so the cleaner is reusable beyond SEC
# filings; the rule number is preserved because it is the citable unit.
RULE_RE = re.compile(
    r"^\s*rule\s+(\d{1,3})\s*([A-Z])?\s*[:.\-—]?\s*(.{0,90})$",
    re.IGNORECASE,
)

ITEM_TITLES = {
    "1":  "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2":  "Properties",
    "3":  "Legal Proceedings",
    "5":  "Market for Registrant's Common Equity",
    "6":  "Selected Financial Data",
    "7":  "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8":  "Financial Statements and Supplementary Data",
    "9A": "Controls and Procedures",
    "10": "Directors and Executive Officers",
    "11": "Executive Compensation",
}

# Boilerplate lines that add no retrievable signal.
BOILERPLATE_RE = re.compile(
    r"^\s*(table of contents|page\s+\d+|\d+\s*$|form\s+10-k\s*$|"
    r"see accompanying notes.*|the accompanying notes are an integral part.*)\s*$",
    re.IGNORECASE,
)

NBSP = "\xa0"


# ------------------------------------------------------------------ dataclass

@dataclass
class Block:
    """One atomic unit of source content, pre-chunking."""
    text: str
    kind: str                    # "text" | "table"
    section: str                 # e.g. "Item 7 - Management's Discussion and Analysis"
    order: int                   # position within the document
    company: str = ""
    filing_date: str = ""
    source_url: str = ""
    n_rows: Optional[int] = None  # tables only
    n_cols: Optional[int] = None  # tables only
    authority: str = ""           # regulation | contract | commitment
    doc_title: str = ""

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------- helpers

def _norm(s: str) -> str:
    """Collapse whitespace and normalise the unicode filings love to emit."""
    s = (s or "").replace(NBSP, " ")
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", " - ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return s.strip()


def _cell_text(td) -> str:
    return _norm(td.get_text(" ", strip=True))


def is_data_table(rows: List[List[str]]) -> bool:
    """
    Distinguish a real data table from a layout table.

    Two kinds of data table matter in this corpus and they look nothing alike:

    * NUMERIC -- compensation schedules, baggage fees, financial statements.
      Detected by a meaningful share of cells containing digits.

    * CATEGORICAL -- the DOT cancellation/delay dashboard is a 9-airline by
      16-commitment matrix whose every cell is "Yes" or "No". It contains no
      digits at all, yet it is the single most valuable table in the corpus
      because it is what makes cross-airline comparison answerable. A purely
      numeric test throws it away, so we also accept regular grids of short,
      repetitive cells.

    Layout tables are excluded either way: they are typically one row, one
    cell, or a single long paragraph of prose.
    """
    non_empty_rows = [r for r in rows if any(c for c in r)]
    if len(non_empty_rows) < 2:
        return False
    width = max((sum(1 for c in r if c) for r in non_empty_rows), default=0)
    if width < 2:
        return False
    cells = [c for r in non_empty_rows for c in r if c]
    if not cells:
        return False

    numeric = sum(1 for c in cells if re.search(r"\d", c))
    if (numeric / len(cells)) >= 0.25:
        return True

    # Categorical grid: at least 3 rows, at least 6 populated cells, cells
    # short enough to be labels rather than prose, and a consistent row width.
    if len(non_empty_rows) >= 3 and len(cells) >= 6:
        lengths = sorted(len(c) for c in cells)
        median_len = lengths[len(lengths) // 2]
        row_widths = [sum(1 for c in r if c) for r in non_empty_rows]
        modal = max(set(row_widths), key=row_widths.count)
        regular = modal >= 2 and row_widths.count(modal) >= max(2, len(row_widths) - 1)
        if median_len <= 40 and regular:
            return True
    return False


SYMBOL_ONLY_RE = re.compile(r"^[\s$€£¥%()\-–—.,:]*$")


def _merge_symbol_cells(cells: List[str]) -> List[str]:
    """
    Filings put the currency symbol in its own <td> for alignment, so a row
    arrives as ['Data Center', '$', '47,525', '$', '15,005', '217%'] while its
    header is ['Segment', 'FY2025', 'FY2024', 'Change']. Left as-is the column
    positions no longer line up, and a model reading the row cannot tell which
    figure belongs to which year. We fold symbol-only cells into the value that
    follows them so the row regains the header's shape.
    """
    out: List[str] = []
    carry = ""
    for c in cells:
        c = c.strip()
        if not c:
            continue
        if SYMBOL_ONLY_RE.match(c) and c not in {"%"}:
            # A lone '(' or '$' prefixes the next number.
            carry += c
            continue
        out.append((carry + c).strip() if carry else c)
        carry = ""
    if carry:
        out.append(carry)
    return out


def table_to_text(rows: List[List[str]]) -> str:
    """
    Render a table as pipe-delimited rows.

    We deliberately do NOT emit markdown alignment rows: the goal is for the
    embedding model and the BM25 index to see 'Revenue | 60,922 | 26,974' as
    one contiguous string, keeping each label glued to its figures.
    """
    kept = []
    for r in rows:
        cells = _merge_symbol_cells(r)
        if cells:
            kept.append(" | ".join(cells))
    return "\n".join(kept)


def _extract_rows(table) -> List[List[str]]:
    rows = []
    for tr in table.find_all("tr"):
        cells = [_cell_text(td) for td in tr.find_all(["td", "th"])]
        rows.append(cells)
    return rows


def _match_item(line: str) -> Optional[str]:
    """SEC filing sections: 'Item 1A. Risk Factors'."""
    m = ITEM_RE.match(line)
    if not m:
        return None
    key = f"{m.group(1)}{(m.group(2) or '').upper()}"
    trailing = _norm(m.group(3) or "").strip(" .-:")
    title = ITEM_TITLES.get(key) or trailing or "Unlabelled"
    return f"Item {key} - {title}"


def _match_rule(line: str) -> Optional[str]:
    """Rule-numbered sections: 'RULE 19: FLIGHT DELAYS'."""
    m = RULE_RE.match(line)
    if not m:
        return None
    key = f"{m.group(1)}{(m.group(2) or '').upper()}"
    title = _norm(m.group(3) or "").strip(" .-:—")
    # Filings-style ALL CAPS headings read badly in a citation.
    if title and title.isupper():
        title = title.title()
    return f"Rule {key}" + (f" - {title}" if title else "")


def _match_none(line: str) -> Optional[str]:
    return None


# Canonical 10-K section titles, for filings that don't number their headings.
#
# Not every company writes "Item 1A. Risk Factors" above its risk section. Some
# file a cross-reference 10-K: the Item numbers live only in an index pointing
# at page numbers, and the body runs under plain titles. Intel is one. For
# those, matching the title itself recovers the structure.
CANONICAL_TITLES = {
    "business": "Item 1 - Business",
    "our business": "Item 1 - Business",
    "risk factors": "Item 1A - Risk Factors",
    "unresolved staff comments": "Item 1B - Unresolved Staff Comments",
    "cybersecurity": "Item 1C - Cybersecurity",
    "properties": "Item 2 - Properties",
    "legal proceedings": "Item 3 - Legal Proceedings",
    "mine safety disclosures": "Item 4 - Mine Safety Disclosures",
    "market for our common stock": "Item 5 - Market for Registrant's Common Equity",
    "management's discussion and analysis": "Item 7 - Management's Discussion and Analysis",
    "operating segment results": "Item 7 - Management's Discussion and Analysis",
    "consolidated results of operations": "Item 7 - Management's Discussion and Analysis",
    "liquidity and capital resources": "Item 7 - Management's Discussion and Analysis",
    "critical accounting estimates": "Item 7 - Management's Discussion and Analysis",
    "quantitative and qualitative disclosures about market risk":
        "Item 7A - Quantitative and Qualitative Disclosures About Market Risk",
    "financial statements and supplementary data":
        "Item 8 - Financial Statements and Supplementary Data",
    "consolidated financial statements":
        "Item 8 - Financial Statements and Supplementary Data",
    "notes to consolidated financial statements":
        "Item 8 - Financial Statements and Supplementary Data",
    "controls and procedures": "Item 9A - Controls and Procedures",
    "information about our executive officers": "Item 10 - Directors and Executive Officers",
    "executive compensation": "Item 11 - Executive Compensation",
}


def _match_title(line: str) -> Optional[str]:
    """Recognise a bare section title, mapped back onto its Item number."""
    key = re.sub(r"[\s:.\-]+$", "", line.strip().lower())
    return CANONICAL_TITLES.get(key)


SECTION_SCHEMES = {
    "item": _match_item,
    "rule": _match_rule,
    "heading": _match_none,   # relies purely on <h1>-<h4> tags
    "title": _match_title,    # bare titles, for cross-reference filings
}

HEADING_TAGS = ("h1", "h2", "h3", "h4")


def _resolve_section(el_name: str, txt: str, scheme: str, current: str):
    """
    Return (section, consumed).

    `consumed` means the text WAS a heading and should not also be emitted as
    body content. Pattern matches win; otherwise a short <h1>-<h4> becomes the
    section, which is what carries structure on ordinary web pages.
    """
    if len(txt) <= 120:
        hit = SECTION_SCHEMES.get(scheme, _match_none)(txt)
        if hit:
            return hit, True
    if el_name in HEADING_TAGS and 3 <= len(txt) <= 120:
        return txt, True
    return current, False


# ---------------------------------------------------------------------- main

def html_to_blocks(
    html: str | bytes,
    company: str = "",
    filing_date: str = "",
    source_url: str = "",
    min_text_chars: int = 60,
    scheme: str = "item",
    authority: str = "",
    doc_title: str = "",
) -> List[Block]:
    """
    Parse one document into an ordered list of clean, section-tagged blocks.

    `scheme` selects how sections are detected: "item" for SEC filings, "rule"
    for contracts of carriage, "heading" for ordinary web pages.
    """
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="ignore")
    html_src = html

    # Some filings are inline-XBRL and technically XML; the HTML parser copes
    # fine and the warning is noise, so it is silenced rather than printed
    # eight times.
    import warnings as _w
    try:
        from bs4 import XMLParsedAsHTMLWarning
        _w.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    except ImportError:
        pass

    soup = BeautifulSoup(html, "lxml")

    for tag in soup(list(DROP_TAGS)):
        tag.decompose()
    # XBRL inline facts are hidden from readers; they wreck text extraction.
    for tag in soup.find_all(attrs={"style": re.compile(r"display\s*:\s*none", re.I)}):
        tag.decompose()
    for tag in soup.find_all(["ix:header", "ix:hidden"]):
        tag.decompose()

    body = soup.body or soup
    blocks: List[Block] = []
    section = doc_title or "Front Matter"
    order = 0

    def push(text: str, kind: str, n_rows=None, n_cols=None):
        nonlocal order
        blocks.append(Block(text=text, kind=kind, section=section, order=order,
                            company=company, filing_date=filing_date,
                            source_url=source_url, n_rows=n_rows, n_cols=n_cols,
                            authority=authority, doc_title=doc_title))
        order += 1

    # Walk only top-level flow elements so nested tables are handled once --
    # with one exception, `td`/`th`, explained below.
    for el in body.find_all(["p", "div", "table", "h1", "h2", "h3", "h4", "h5",
                             "li", "dt", "dd", "td", "th"],
                            recursive=True):

        # Some filers lay their whole document out in tables, so the Item
        # headings that mark sections live in <td> cells rather than in
        # paragraphs. Skipping everything inside a table -- which is otherwise
        # right, so a nested table is not processed twice -- loses every
        # section boundary in those documents, and every chunk ends up tagged
        # "Front Matter". Intel's filing is one of these: 17,000 <td> elements
        # and not a single heading outside one.
        #
        # So cells get inspected for a section heading, but are NEVER emitted
        # as content: the enclosing <table> still owns the text, and letting a
        # cell speak twice would duplicate it.
        if el.name in ("td", "th"):
            cell = _norm(el.get_text(" ", strip=True))
            # Short only. A long cell is data, and a table-of-contents row
            # ("Item 1A. Risk Factors Pages 37 - 51") is not a real heading.
            if 3 <= len(cell) <= 60:
                sec, consumed = _resolve_section("td", cell, scheme, section)
                if consumed:
                    section = sec
            continue

        if el.find_parent("table") is not None:
            continue  # handled when we reach the outer table

        if el.name == "table":
            rows = _extract_rows(el)
            if is_data_table(rows):
                txt = table_to_text(rows)
                if txt:
                    width = max((len([c for c in r if c]) for r in rows), default=0)
                    push(txt, "table", n_rows=len(rows), n_cols=width)
            else:
                # Layout table: salvage its text, it often holds headings.
                txt = _norm(el.get_text(" ", strip=True))
                sec, consumed = _resolve_section(el.name, txt, scheme, section)
                if consumed:
                    section = sec
                elif len(txt) >= min_text_chars and not BOILERPLATE_RE.match(txt):
                    push(txt, "text")
            continue

        txt = _norm(el.get_text(" ", strip=True))
        if not txt or BOILERPLATE_RE.match(txt):
            continue

        sec, consumed = _resolve_section(el.name, txt, scheme, section)
        if consumed:
            section = sec
            continue

        if len(txt) >= min_text_chars:
            # Skip if an ancestor already contributed this exact text.
            if blocks and blocks[-1].kind == "text" and txt in blocks[-1].text:
                continue
            push(txt, "text")

    blocks = _dedupe(blocks)

    # If Item-numbered headings found essentially nothing, this is a
    # cross-reference filing -- the Item numbers live in an index, not above
    # the sections. Retry reading bare titles instead. Guarded so the seven
    # filings that work normally are untouched, and it only ever runs once.
    if scheme == "item" and len({b.section for b in blocks}) <= 2:
        retry = html_to_blocks(html_src, company=company, filing_date=filing_date,
                               source_url=source_url, min_text_chars=min_text_chars,
                               scheme="title", authority=authority, doc_title=doc_title)
        if len({b.section for b in retry}) > len({b.section for b in blocks}):
            return retry

    return blocks


def _dedupe(blocks: List[Block]) -> List[Block]:
    """Nested divs cause the same paragraph to be emitted repeatedly."""
    seen = set()
    out = []
    for b in blocks:
        key = (b.kind, b.text[:300])
        if key in seen:
            continue
        seen.add(key)
        b.order = len(out)
        out.append(b)
    return out


def corpus_stats(blocks: List[Block]) -> dict:
    tables = [b for b in blocks if b.kind == "table"]
    texts = [b for b in blocks if b.kind == "text"]
    return {
        "blocks": len(blocks),
        "text_blocks": len(texts),
        "table_blocks": len(tables),
        "sections": sorted({b.section for b in blocks}),
        "chars": sum(len(b.text) for b in blocks),
        "table_chars": sum(len(b.text) for b in tables),
    }
