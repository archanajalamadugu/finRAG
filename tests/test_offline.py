"""
Offline test suite. No network, no API key, no LangChain required.

Covers the logic that is easy to get subtly wrong and expensive to debug on
Saturday: sentence splitting on filing prose, table classification, currency
cell realignment, section tagging, and the behavioural difference between the
two chunking strategies.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.clean import (html_to_blocks, is_data_table, table_to_text,
                       _merge_symbol_cells, corpus_stats)
from extras.chunkers import split_sentences, chunk_fixed, chunk_semantic, chunk_stats
from tests.mock_embed import MockEmbedder

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "mini_10k.html"

_passed, _failed = 0, 0

def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def test_sentences():
    s = ("Acme Inc. reported revenue of $1.5 billion in the U.S. market. "
         "Growth was 12.4% year over year. Mr. Chen noted supply constraints. "
         "See Note No. 3 for details.")
    out = split_sentences(s)
    check("sentence count on abbreviation-heavy prose", len(out) == 4, f"got {len(out)}")
    check("decimal not split", all("$1.5 billion" in x or "$1.5" not in x for x in out))
    check("percentage not split", any("12.4%" in x for x in out))
    check("'Inc.' not treated as boundary", out[0].startswith("Acme Inc. reported"))


def test_tables():
    layout = [["", "Logo placeholder for layout only"]]
    data = [["Segment", "FY2025", "FY2024"], ["Revenue", "60,922", "26,974"],
            ["Net income", "29,760", "4,368"]]
    check("layout table rejected", not is_data_table(layout))
    check("data table accepted", is_data_table(data))
    check("currency cells folded forward",
          _merge_symbol_cells(["Data Center", "$", "47,525", "$", "15,005"])
          == ["Data Center", "$47,525", "$15,005"])
    check("negatives keep their paren",
          _merge_symbol_cells(["Loss", "(", "1,204", ")"])[1].startswith("("))
    txt = table_to_text(data)
    check("row renders pipe-delimited", "Revenue | 60,922 | 26,974" in txt)


def test_clean():
    blocks = html_to_blocks(FIXTURE.read_text(), company="AcmeSemi",
                            filing_date="2025-02-26")
    st = corpus_stats(blocks)
    secs = st["sections"]
    check("all four Item sections detected", len(secs) == 4, str(secs))
    check("MD&A section labelled", any("Item 7" in s for s in secs))
    check("both data tables kept", st["table_blocks"] == 2, str(st["table_blocks"]))
    joined = " ".join(b.text for b in blocks)
    check("hidden XBRL dropped", "XBRL hidden fact" not in joined)
    check("script contents dropped", "var a=1" not in joined)
    check("boilerplate dropped", "Table of Contents" not in joined)
    check("page number dropped", not any(b.text.strip() == "42" for b in blocks))
    check("notes boilerplate dropped", "integral part" not in joined)
    check("layout/logo table dropped", "Logo placeholder" not in joined)
    return blocks


def test_chunkers(blocks):
    emb = MockEmbedder().fit([b.text for b in blocks])
    fx  = chunk_fixed(blocks, chunk_tokens=128, overlap_tokens=16, atomic_tables=False)
    fxa = chunk_fixed(blocks, chunk_tokens=128, overlap_tokens=16, atomic_tables=True)
    sem = chunk_semantic(blocks, emb.embed_documents, breakpoint_percentile=75,
                         max_tokens=200, min_tokens=20)

    def intact(cs):
        return any("Total revenue | $60,922" in c.text and "Segment | FY2025" in c.text
                   for c in cs)

    check("naive fixed splits the table (baseline failure reproduced)", not intact(fx))
    check("fixed + atomic keeps the table whole", intact(fxa))
    check("semantic keeps the table whole", intact(sem))
    check("semantic respects the token cap",
          all(c.n_tokens <= 260 for c in sem),
          str(max(c.n_tokens for c in sem)))
    check("no sliver chunks survive",
          all(c.n_tokens >= 20 for c in sem),
          str(min(c.n_tokens for c in sem)))
    check("every chunk carries a section", all(c.section for c in sem))
    check("every chunk carries company metadata", all(c.company == "AcmeSemi" for c in sem))
    check("citations carry company and fiscal date",
          sem[0].citation().startswith("[AcmeSemi 2025-02-26, Item 1"),
          sem[0].citation())
    check("chunk ids unique", len({c.chunk_id for c in sem}) == len(sem))
    check("strategies produce different boundaries", len(sem) != len(fx))
    check("stats report cleanly", chunk_stats(sem)["n_chunks"] == len(sem))


def test_rule_scheme():
    """Contract-of-carriage sections are numbered Rules, not Items."""
    html = (pathlib.Path(__file__).parent / "fixtures" / "mini_coc.html").read_text()
    blocks = html_to_blocks(html, company="Delta", scheme="rule",
                            authority="contract",
                            doc_title="Contract of Carriage")
    secs = {b.section for b in blocks}
    check("Rule 19 detected", any(s.startswith("Rule 19") for s in secs), str(secs))
    check("Rule 20 detected", any(s.startswith("Rule 20") for s in secs))
    check("rule title case-normalised",
          any("Flight Delays" in s for s in secs), str(secs))
    check("ALL-CAPS heading not emitted as body text",
          not any(b.text.startswith("RULE 19") for b in blocks))
    check("compensation table captured",
          any(b.kind == "table" and "400% of fare" in b.text for b in blocks))
    tbl = [b for b in blocks if b.kind == "table"][0]
    check("dollar cells folded in compensation table",
          "$0" in tbl.text, tbl.text[:120])
    check("authority tier carried", all(b.authority == "contract" for b in blocks))
    hotel = [b for b in blocks if "hotel accommodation" in b.text]
    check("hotel provision sits under Rule 19",
          bool(hotel) and hotel[0].section.startswith("Rule 19"),
          hotel[0].section if hotel else "not found")


def test_heading_scheme():
    """Ordinary web pages carry structure in <h1>-<h4>, not numbered patterns."""
    html = (pathlib.Path(__file__).parent / "fixtures" / "mini_dot.html").read_text()
    blocks = html_to_blocks(html, company="US DOT", scheme="heading",
                            authority="regulation",
                            doc_title="Cancellation and Delay Dashboard")
    secs = {b.section for b in blocks}
    check("h2 becomes a section", "Controllable Cancellations" in secs, str(secs))
    check("second h2 becomes a section",
          any("3 Hours" in s for s in secs), str(secs))
    tables = [b for b in blocks if b.kind == "table"]
    check("both dashboard matrices captured", len(tables) == 2, str(len(tables)))
    check("airline rows kept intact",
          any("Frontier | Yes | No | Yes | No" in b.text for b in tables))
    check("delay table tagged to its own section",
          tables[1].section != tables[0].section,
          f"{tables[0].section} / {tables[1].section}")
    check("regulation tier carried", all(b.authority == "regulation" for b in blocks))


def test_citation_tiers():
    from extras.chunkers import Chunk
    c = Chunk(text="x", chunk_id="c1", strategy="s", section="Rule 19 - Flight Delays",
              company="Delta", filing_date="", kind="text",
              authority="contract", doc_title="Contract of Carriage")
    check("tiered citation still works for non-filing sources",
          c.citation() == "[Delta, Rule 19 - Flight Delays (binding contract)]",
          c.citation())
    d = Chunk(text="x", chunk_id="c2", strategy="s", section="Refunds",
              company="US DOT", filing_date="", kind="text",
              authority="regulation", doc_title="Refunds")
    check("regulation tier appears in citation", "regulation" in d.citation(), d.citation())
    e = Chunk(text="x", chunk_id="c3", strategy="s", section="Customer Service Plan",
              company="American", filing_date="", kind="text",
              authority="commitment", doc_title="Customer Service Plan")
    check("commitment citation flags it as voluntary",
          "voluntary commitment" in e.citation(), e.citation())


def test_corpus_definition():
    """The active corpus is the SEC filings one."""
    import src.fetch_filings as ff

    check("eight companies in the corpus", len(ff.TICKERS) == 8, str(ff.TICKERS))
    check("every ticker has a display name",
          all(t in ff.NICE_NAMES for t in ff.TICKERS))
    check("tickers are unique", len(set(ff.TICKERS)) == len(ff.TICKERS))

    # Ticker -> CIK resolution, against a stubbed copy of the SEC's mapping file
    # so the test needs no network.
    import json as _json
    fake = _json.dumps({
        "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
        "1": {"cik_str": 2488,    "ticker": "AMD",  "title": "ADVANCED MICRO DEVICES"},
        "2": {"cik_str": 320193,  "ticker": "AAPL", "title": "Apple Inc."},
    }).encode()
    real_get = ff.get
    ff.get = lambda url, timeout=60: fake
    try:
        got = ff.ticker_to_cik(["NVDA", "amd"])
        check("tickers resolve to CIKs", got == {"NVDA": "1045810", "AMD": "2488"}, str(got))
        check("lookup is case-insensitive", "AMD" in got)
        check("CIKs are digits, zero-paddable to 10",
              all(c.isdigit() and len(c) <= 10 for c in got.values()))
        try:
            ff.ticker_to_cik(["NOTAREALTICKER"])
            check("unknown ticker raises rather than skipping", False)
        except ValueError:
            check("unknown ticker raises rather than skipping", True)
        names = ff.resolve_companies(["NVDA"], ff.NICE_NAMES)
        check("display names replace SEC's shouty titles", names == {"NVIDIA": "1045810"}, str(names))
    finally:
        ff.get = real_get
    # The evaluation question set lives in the notebook (section 12), next to
    # the code that runs it, rather than in a separate file that can drift out
    # of sync. Nothing to assert here offline.

if __name__ == "__main__":
    print("\nsentences"); test_sentences()
    print("\ntables");    test_tables()
    print("\ncleaning");  blocks = test_clean()
    print("\nchunking");  test_chunkers(blocks)
    print("\nrule scheme (generalised tagger)"); test_rule_scheme()
    print("\nheading scheme (generalised tagger)"); test_heading_scheme()
    print("\ncitation tiers");              test_citation_tiers()
    print("\ncorpus definition");           test_corpus_definition()
    print(f"\n{_passed} passed, {_failed} failed\n")
    sys.exit(1 if _failed else 0)
