"""
Read the PDF's bookmarks (the outline in the left sidebar) and propose which pages to skip.

Usage:
  python scripts/toc_ranges.py data/raw/oppenheim.pdf

Prints (1) the bookmark list with PDF page numbers, and (2) a suggested --skip-pages argument
that removes front matter, each chapter's problem sets, and the back matter (appendix,
bibliography, answers, index). Paste the output back so the ranges can be checked before
the full run.
"""
import re
import sys

try:
    import pymupdf as fitz
except ImportError:
    import fitz

PROBLEM_RE = re.compile(r"problem", re.I)
BACK_RE = re.compile(r"^(appendix|bibliography|answers|index)\b", re.I)
CHAPTER_RE = re.compile(r"^\s*(chapter\s+)?\d+\b", re.I)


def merge(ranges):
    out = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def propose(toc, n_pages):
    """toc: list of [level, title, page(1-based)]. Returns (ranges, notes)."""
    ranges, notes = [], []
    toc = [t for t in toc if t[2] > 0]

    # front matter: everything before the first numbered chapter
    first_ch = next((t for t in toc if CHAPTER_RE.match(t[1])), None)
    if first_ch and first_ch[2] > 1:
        ranges.append((1, first_ch[2] - 1))
        notes.append(f"front matter: 1-{first_ch[2] - 1} (before '{first_ch[1]}')")

    # problem sets: from a 'Problems' bookmark to the next bookmark that is not a problem set
    for i, (lvl, title, pg) in enumerate(toc):
        if not PROBLEM_RE.search(title):
            continue
        end = n_pages
        for lvl2, title2, pg2 in toc[i + 1:]:
            if lvl2 <= lvl and not PROBLEM_RE.search(title2):
                end = max(pg, pg2 - 1)
                break
        if end - pg + 1 > 80:
            notes.append(f"WARNING: problems '{title}' would skip {pg}-{end} ({end - pg + 1} pages) - "
                         "probably a missing chapter bookmark; NOT included")
            continue
        ranges.append((pg, end))
        notes.append(f"problems: {pg}-{end} ('{title}')")

    # back matter: appendix / bibliography / answers / index, to the end of the book
    back = [t for t in toc if t[0] == 1 and BACK_RE.match(t[1])]
    if back:
        start = min(t[2] for t in back)
        ranges.append((start, n_pages))
        notes.append(f"back matter: {start}-{n_pages} (from '{min(back, key=lambda t: t[2])[1]}')")
    return merge(ranges), notes


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    doc = fitz.open(sys.argv[1])
    toc = doc.get_toc(simple=True)
    print(f"{len(doc)} pages, {len(toc)} bookmarks\n")
    if not toc:
        sys.exit("No bookmarks in this PDF; skip ranges have to be set by hand.")
    for lvl, title, pg in toc:
        if lvl <= 3:
            print(f"{'  ' * (lvl - 1)}{title}  -> PDF page {pg}")
    ranges, notes = propose(toc, len(doc))
    print("\nProposed skips:")
    for n in notes:
        print("  ", n)
    print("\nSuggested argument:")
    print("  --skip-pages " + " ".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges))