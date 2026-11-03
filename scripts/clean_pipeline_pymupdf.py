"""
Signals & Systems PDF cleaning pipeline (PyMuPDF) -- built for SCANNED textbooks
with an OCR text layer (tested against the layout of Oppenheim/Willsky/Nawab).

Based on Riya Pawar's font profiling idea (Algorithm 1), with changes for
scanned books, where a single exact font size does not work:
  * body text is identified by FONT FAMILY + a size band, line by line
    (examples are set smaller than main text; OCR makes sizes drift)
  * figure and caption regions are found FIRST (from the caption position) and
    everything inside them is kept out of the text (panel labels, captions)
  * a line with real words stays text even if it is math-heavy
  * hidden-OCR math fragments become <eq_pN_k> markers, not silent holes
  * display equations become markers too; both are saved to equations.json
  * figures are cropped by REGION (the page is one big scan image), saved as
    PDF + PNG with the caption text; several captions per page are handled

Outputs (in --out):
  clean_text.txt / clean_text.json   body text with <eq_...> markers and <fig_...: caption> markers
  equations.json                     marker id, page, kind (inline/display), OCR text
  captions.json + figures/           figure crops (pdf + png) with caption text
  chunks/ + chunk_manifest.json      LLM-ready chunks

Usage:
  pip install pymupdf
  python clean_pipeline_pymupdf.py oppenheim.pdf --out output/ \
      --skip-pages 1-31 960-987 --page-offset 31

  # try a few pages first:
  python clean_pipeline_pymupdf.py oppenheim.pdf --out test/ --only-pages 40-42
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter

try:
    import pymupdf as fitz  # newer name, no deprecation warning
except ImportError:
    import fitz  # PyMuPDF

CAPTION_RE = re.compile(r"^(Figure|Fig\.|Table)\s*[\dlI]+[.\dlI]*", re.I)   # Oppenheim prints TABLE 3.1 in capitals
ROW_GAP = 14  # max gap (pt) between caption rows before they count as separate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def font_key(font):
    """Short family key: drop subset prefix (ABCDEF+), lowercase, first 5 chars.
    Groups Times-Roman / Times-Italic / TimesNewRomanPSMT together."""
    return font.split("+")[-1].lower()[:5]


def is_hidden(font):
    """Hidden OCR text layer (e.g. 'HiddenHorzOCR')."""
    return "ocr" in font.lower()


def parse_pages(tokens):
    """['1-31', '960', '970-975'] -> {1..31, 960, 970..975}"""
    out = set()
    for t in tokens or []:
        if "-" in t:
            a, b = t.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(t))
    return out


_PUNCT = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "−": "-", "‘": "'", "’": "'", "“": '"', "”": '"',
}


def normalize(text):
    """Light normalization. Unlike the paper's ASCII sanitization this KEEPS
    non-ASCII letters (omega, pi, tau...), which carry meaning in Signals text;
    it only unifies dashes/quotes and expands ligatures."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(_PUNCT.get(c, c) for c in text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def gap_space(prev, cur):
    """' ' if two neighbouring spans are visibly apart but neither carries a space
    (the OCR layer often drops the space glyph: 'the' + 'nonzero')."""
    pb, cb = prev.get("bbox"), cur.get("bbox")
    if not pb or not cb or not prev["text"] or not cur["text"]:
        return ""
    if prev["text"][-1].isspace() or cur["text"][0].isspace():
        return ""
    return " " if cb[0] - pb[2] > 0.18 * max(cur["size"], 1) else ""


def is_word_piece(prev, hidden, nxt, raw):
    """A short hidden-layer fragment inside a word ('bra' + 'n' + 'ches', or 'the ' + 'ge' + 'neral'):
    the OCR layer swallowed some letters. Letters only, 1-4 of them, touching the visible text."""
    if not re.fullmatch(r"[A-Za-z]{1,4}", raw) or not nxt["text"][:1].isalpha():
        return False
    pt = prev["text"]
    if not (pt[-1:].isalpha() or pt[-1:].isspace()):
        return False
    hb, nb, pb = hidden[0].get("bbox"), nxt.get("bbox"), prev.get("bbox")
    if not (hb and nb and pb):
        return True
    lim = 0.25 * max(nxt["size"], 1)
    touches_next = nb[0] - hidden[-1].get("bbox", hb)[2] < lim
    touches_prev = hb[0] - pb[2] < lim
    return touches_next and (touches_prev if pt[-1:].isalpha() else True)


def line_text(spans):
    """Line text with hidden-OCR spans shown as [math] (used for captions)."""
    out = []
    for i, s in enumerate(spans):
        if i and not is_hidden(s["font"]) and not is_hidden(spans[i - 1]["font"]):
            out.append(gap_space(spans[i - 1], s))
        out.append(" [math] " if is_hidden(s["font"]) else s["text"])
    return "".join(out).strip()


# ---------------------------------------------------------------------------
# Body-text detection
# ---------------------------------------------------------------------------

def detect_body_params(doc, sample=200):
    """Find the dominant (non-hidden) font family and its most common size,
    sampling up to `sample` pages. Returns family key, mode size, size band."""
    fam = Counter()
    sizes = {}
    step = max(1, len(doc) // sample)
    for i in range(0, len(doc), step):
        for b in doc[i].get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    n = len(s["text"].strip())
                    if not n or is_hidden(s["font"]):
                        continue
                    k = font_key(s["font"])
                    fam[k] += n
                    sizes.setdefault(k, Counter())[round(s["size"] * 2) / 2] += n
    if not fam:
        raise ValueError("No text found. Is this PDF image-only (no OCR layer)?")
    family = fam.most_common(1)[0][0]
    mode = sizes[family].most_common(1)[0][0]
    print(f"[detect] font families by chars: {fam.most_common(5)}")
    print(f"[detect] sizes in '{family}': {sizes[family].most_common(6)}")
    return {"family": family, "mode": mode,
            "min": round(0.84 * mode, 1), "max": round(1.12 * mode, 1),
            "frac": 0.6, "prose_words": 2}


def page_lines(page, p):
    """Text lines in stream (reading) order, each flagged is_body: at least
    `frac` of its characters are body-family text within the size band."""
    lines = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            spans = line["spans"]
            total = sum(len(s["text"]) for s in spans) or 1
            body = sum(len(s["text"]) for s in spans
                       if font_key(s["font"]) == p["family"]
                       and p["min"] <= s["size"] <= p["max"])
            lines.append({"bbox": fitz.Rect(line["bbox"]), "spans": spans,
                          "is_body": body / total >= p["frac"]})
    return lines


# ---------------------------------------------------------------------------
# Figures and captions (found first, so their lines can be kept out of the text)
# ---------------------------------------------------------------------------

def table_bottom(page, x0, x1, y_top, y_limit, min_rules=2):
    """Where a table ends: the last long horizontal rule below its title. A rule is a row with a
    solid dark run covering at least half the table width (text rows never do; dashed
    separators don't count). Returns None if fewer than `min_rules` rules are found."""
    if y_limit - y_top < 20:
        return None
    try:
        reg = fitz.Rect(x0, y_top, x1, y_limit)
        pix = page.get_pixmap(clip=reg, dpi=72, colorspace=fitz.csGRAY)
    except Exception:
        return None
    w, h, data = pix.width, pix.height, pix.samples
    need = 0.5 * w
    rules = []
    for y in range(h):
        row = data[y * w:(y + 1) * w]
        nxt = data[(y + 1) * w:(y + 2) * w] if y + 1 < h else row
        run = best = 0
        for a, b in zip(row, nxt):                      # a row OR the next one: a slightly skewed scan
            run = run + 1 if (a < 190 or b < 190) else 0   # splits one rule across two pixel rows
            if run > best:
                best = run
        if best >= need:
            rules.append(y)
    if len(rules) < min_rules:
        return None
    return y_top + (rules[-1] + 1) * (reg.height / h)


LABEL_RE = re.compile(r"^(\(?[a-e]\)?|[\d\-\u2022.\s/nmkt]{1,6})$")


def is_label_frag(line):
    """Tiny figure labels (axis ticks like '0', 'n', panel tags like '(b)', bullets)."""
    return bool(LABEL_RE.match(line_text(line["spans"]).strip())) and len(line_text(line["spans"]).strip()) <= 6


def find_figures(page, p):
    """Find every caption on the page (figures can sit side by side, as in a 2x2
    grid). Each caption owns a column: from its own left edge to the next caption
    on the same row. Returns a list of dicts: kind, caption, rect (figure crop,
    None for tables) and skip (x0, y0, x1, y1) region, figure + caption, whose
    lines must stay out of the body text."""
    W, H = page.rect.width, page.rect.height
    lines = sorted(page_lines(page, p), key=lambda l: l["bbox"].y0)

    def starts_like_caption(l):
        # real captions are set in Helvetica; a Times line that merely begins
        # with "Figure 1.1 shows..." is prose
        first = next((s for s in l["spans"] if s["text"].strip() and not is_hidden(s["font"])), None)
        return (not l["is_body"] and first is not None
                and font_key(first["font"]) != p["family"]
                and CAPTION_RE.match(line_text(l["spans"])))

    caps = [l for l in lines if starts_like_caption(l)]

    # column extent of each caption
    def col_of(cap):
        right = [c["bbox"].x0 for c in caps
                 if c is not cap and abs(c["bbox"].y0 - cap["bbox"].y0) < 60
                 and c["bbox"].x0 > cap["bbox"].x0 + 30]
        left = [c["bbox"].x0 for c in caps
                if c is not cap and abs(c["bbox"].y0 - cap["bbox"].y0) < 60
                and c["bbox"].x0 < cap["bbox"].x0 - 30]
        x1 = (min(right) - 6) if right else 0.96 * W
        x0 = (cap["bbox"].x0 - 6) if left else 0.10 * W
        return x0, x1

    results, used = [], set()
    for cap in caps:
        if id(cap) in used:
            continue
        cx0, cx1 = col_of(cap)
        kind = "table" if line_text(cap["spans"]).lower().startswith("table") else "figure"
        # lines in this column (by horizontal centre), at or below the caption start
        cand = [l for l in lines
                if l is cap or (l["bbox"].y0 >= cap["bbox"].y0 - 8
                                and l["bbox"].x0 >= cap["bbox"].x0 - 4          # caption block is left-aligned
                                and cx0 <= (l["bbox"].x0 + l["bbox"].x1) / 2 < cx1
                                and not is_label_frag(l))]
        cand.sort(key=lambda f: (f["bbox"].y0 + f["bbox"].y1) / 2)
        rows, row, row_yc = [], [], None
        for f in cand:
            yc = (f["bbox"].y0 + f["bbox"].y1) / 2
            if row and abs(yc - row_yc) > 5:
                rows.append(row)
                row = []
            if not row:
                row_yc = yc
            row.append(f)
        if row:
            rows.append(row)

        caps_ids = {id(c) for c in caps}
        kept = []
        for r in rows:
            if kind == "table" and kept:
                break                                  # a table's rows are not part of its title
            has_cap_font = any(not f["is_body"] for f in r)
            if kept:
                gap = min(f["bbox"].y0 for f in r) - max(f["bbox"].y1 for f in kept[-1])
                if any(id(f) in caps_ids and f is not cap for f in r):
                    break                              # a new caption starts here
                if gap > ROW_GAP:
                    break                              # big gap: next figure's labels
                if not has_cap_font:
                    # body-font row: only a short tail line touching the caption
                    # (e.g. a math ending set in Times) belongs to it
                    short = sum(len(line_text(f["spans"])) for f in r) < 60
                    if not (short and gap <= 2):
                        break
            elif not has_cap_font:
                continue
            kept.append(r)
        if not kept:
            continue
        used.update(id(f) for r in kept for f in r)

        cap_top = min(f["bbox"].y0 for r in kept for f in r)
        cap_bottom = max(f["bbox"].y1 for r in kept for f in r)
        caption = " ".join(" ".join(line_text(f["spans"]) for f in sorted(r, key=lambda f: f["bbox"].x0))
                           for r in kept)
        # OCR fixes in the label: "Figure 1 .1 O" -> "Figure 1.10", "Figure 1. 13" -> "Figure 1.13"
        caption = re.sub(r"^(Figure|Fig\.|Table)\s*(\d+)\s*\.\s*(\d+)\s*[Oo]\b", r"\1 \2.\g<3>0", caption)
        caption = re.sub(r"^(Figure|Fig\.|Table)\s*(\d+)\s*\.\s*(\d+)", r"\1 \2.\3", caption, flags=re.I)
        label = (re.match(r"^((?:Figure|Fig\.|Table) \d+\.\d+)", caption, re.I) or [None, None])[1]
        caption = re.sub(r"([A-Za-z]{2})- ([a-z])", r"\1\2", caption)   # line-break hyphenation
        caption = normalize(caption)

        rect, skip = None, (cx0, cap_top, cx1, cap_bottom)
        if kind == "table":
            # the table runs from its title down to its last horizontal rule (or, failing that,
            # to the next line of prose, or the bottom margin)
            y_top = cap_top - 3
            y_limit = 0.94 * H
            prose = [l["bbox"].y0 - 2 for l in lines if l["is_body"] and l["bbox"].y0 > cap_bottom
                     and len(line_text(l["spans"])) > 60]
            y_max = min([y_limit] + prose)                      # a table never runs into the next paragraph
            rule_bottom = table_bottom(page, cx0, cx1, y_top, y_max)
            inside = [l["bbox"].y1 for l in lines
                      if l["bbox"].y0 >= y_top and l["bbox"].y1 <= y_max
                      and cx0 <= (l["bbox"].x0 + l["bbox"].x1) / 2 <= cx1]
            text_bottom = (max(inside) + 4) if inside else cap_bottom
            bottom = min(y_max, max(rule_bottom or 0, text_bottom, cap_bottom))
            rect = fitz.Rect(cx0, y_top, cx1, bottom)
            skip = (cx0, y_top, cx1, bottom)
        if kind == "figure":
            # top of the figure: header margin, body text above, or the caption
            # of whatever sits above in the same column
            # prose above the figure: a body line that is wide like a paragraph line (a label set
            # in Times inside the figure is narrow, even when it is longer than 12 characters)
            above = [l for l in lines if l["is_body"]
                     and len(line_text(l["spans"])) > 12 and l["bbox"].y1 <= cap_top
                     and l["bbox"].width >= 0.30 * W]
            col_above = [r["skip"][3] for r in results
                         if r["skip"][2] > cx0 and r["skip"][0] < cx1 and r["skip"][3] < cap_top]
            top = max([0.085 * H] + [l["bbox"].y1 for l in above] + col_above)   # 0.085H clears the running header
            rect = fitz.Rect(cx0, top + 2, cx1, cap_top - 2)
            skip = (cx0, top, cx1, cap_bottom)
        results.append({"kind": kind, "label": label, "rect": rect, "caption": caption, "skip": skip,
                        "cap_boxes": [tuple(f["bbox"]) for r in kept for f in r], "cap_bottom": cap_bottom,
                        "cap_top": cap_top, "col": (cx0, cx1)})
    return results


def paint_white(data, w, h, strip, boxes, pad=2):
    """Paint page-coordinate boxes white in a gray pixel buffer covering `strip`."""
    sx, sy = strip.width / w, strip.height / h
    for bx0, by0, bx1, by1 in boxes:
        xa, xb = max(0, int((bx0 - pad - strip.x0) / sx)), min(w, int((bx1 + pad - strip.x0) / sx) + 1)
        ya, yb = max(0, int((by0 - 1 - strip.y0) / sy)), min(h, int((by1 + 1 - strip.y0) / sy) + 1)
        for y in range(ya, yb):
            if xb > xa:
                data[y * w + xa:y * w + xb] = b"\xff" * (xb - xa)


def trim_to_ink(page, rect, pad=4.0, thresh=190, min_dark=2, mask=()):
    """Shrink `rect` to the box that actually contains ink (the page is a scan, so
    text positions alone give loose or clipped crops). Renders at 72 dpi (1 px = 1 pt)."""
    pix = page.get_pixmap(clip=rect, dpi=72, colorspace=fitz.csGRAY)
    w, h = pix.width, pix.height
    data = bytearray(pix.samples)
    if w < 5 or h < 5:
        return rect
    paint_white(data, w, h, rect, mask)                # ignore caption text when measuring
    rows = [sum(1 for v in data[y * w:(y + 1) * w] if v < thresh) for y in range(h)]
    cols = [0] * w
    for y in range(h):
        row = data[y * w:(y + 1) * w]
        for x, v in enumerate(row):
            if v < thresh:
                cols[x] += 1
    ys = [y for y, c in enumerate(rows) if c >= min_dark]
    xs = [x for x, c in enumerate(cols) if c >= min_dark]
    if not ys or not xs:
        return rect
    sx, sy = rect.width / w, rect.height / h
    r = fitz.Rect(rect.x0 + xs[0] * sx - pad, rect.y0 + ys[0] * sy - pad,
                  rect.x0 + (xs[-1] + 1) * sx + pad, rect.y0 + (ys[-1] + 1) * sy + pad)
    return r & rect                                   # never grow past the search region


def extend_to_figure_bottom(page, rect, cap_boxes, cap_bottom, thresh=190, min_dark=2):
    """The caption often starts on the same rows as the figure's axis and tick labels
    (0, n, m ...). Look in the caption's own rows, with the caption text painted out,
    and extend the crop down to any ink that remains (it belongs to the figure)."""
    y0, y1 = rect.y1, min(cap_bottom + 2, page.rect.height)
    if y1 - y0 < 3:
        return rect
    strip = fitz.Rect(rect.x0, y0, rect.x1, y1)
    pix = page.get_pixmap(clip=strip, dpi=72, colorspace=fitz.csGRAY)
    w, h = pix.width, pix.height
    data = bytearray(pix.samples)
    sy = strip.height / h
    paint_white(data, w, h, strip, cap_boxes)                  # paint the caption text white
    last = -1
    for y in range(h):
        if sum(1 for v in data[y * w:(y + 1) * w] if v < thresh) >= min_dark:
            last = y
    if last < 0:
        return rect
    return fitz.Rect(rect.x0, rect.y0, rect.x1, min(y1, y0 + (last + 1) * sy + 2))


def save_figure(doc, page, rect, base, trim=True, mask=()):
    """Save the figure as PDF + PNG. Caption text that falls inside the crop (the caption
    often sits beside the figure's lower part) is painted white; it is stored as text anyway."""
    if trim:
        rect = trim_to_ink(page, rect, mask=mask)
    crop = fitz.open()
    cp = crop.new_page(width=rect.width, height=rect.height)
    cp.show_pdf_page(cp.rect, doc, page.number, clip=rect)
    for bx0, by0, bx1, by1 in mask:
        box = fitz.Rect(bx0 - 2 - rect.x0, by0 - 1 - rect.y0, bx1 + 2 - rect.x0, by1 + 1 - rect.y0) & cp.rect
        if box.width > 0 and box.height > 0:
            cp.draw_rect(box, color=(1, 1, 1), fill=(1, 1, 1), width=0)
    crop.save(base + ".pdf")
    cp.get_pixmap(dpi=200).save(base + ".png")
    crop.close()
    return rect


# ---------------------------------------------------------------------------
# Stage 1: clean text with equation markers
# ---------------------------------------------------------------------------

def clean_page(page, eqs, p, skip=(), captions=()):
    """captions: [(y_top, '<fig_pN_k: caption>')] - inserted into the text where the figure/table sits."""
    pending = sorted(captions, key=lambda c: c[0])
    page_num = page.number + 1
    H = page.rect.height

    def add_eq(text, kind):
        eq_id = f"eq_p{page_num}_{len(eqs)}"
        eqs.append({"id": eq_id, "page": page_num, "kind": kind, "ocr_text": text.strip()})
        return f"<{eq_id}>"

    def hidden_to_text(raw):
        # keep real words (2+ letters) the OCR layer swallowed; marker for the rest
        pieces = re.split(r"([A-Za-z]{2,}[,.]?)", raw)
        parts = []
        for i, piece in enumerate(pieces):
            if i % 2 == 1:
                parts.append(piece)
            elif re.search(r"[^\s.,]", piece):
                parts.append(add_eq(piece, "inline"))
        return " ".join(parts)

    def render(spans):
        parts, hidden, prev = [], [], None
        for s in spans:
            if is_hidden(s["font"]):
                hidden.append(s)
            else:
                if hidden:
                    raw = "".join(h["text"] for h in hidden).strip()
                    if prev is not None and is_word_piece(prev, hidden, s, raw):
                        parts.append(raw)                   # 'bra' + [n] + 'ches' -> 'branches'
                    else:
                        parts.append(hidden_to_text("".join(h["text"] for h in hidden)) + " ")
                    hidden = []
                elif prev is not None:
                    parts.append(gap_space(prev, s))
                parts.append(s["text"])
                prev = s
        if hidden:
            parts.append(" " + hidden_to_text("".join(h["text"] for h in hidden)))
        return "".join(parts).strip()

    out, nums, hids = [], [], []
    last = None                                        # (marker, eq record) of the open display equation
    NUM_RE = re.compile(r"^\(?\s*\d+\s*\.\s*\d+\s*\)?$")
    for line in page_lines(page, p):
        spans = line["spans"]
        while pending and pending[0][0] <= line["bbox"].y0 + 1:   # figure/table starts above this line
            out.append(pending.pop(0)[1])
            last = None
        xc = (line["bbox"].x0 + line["bbox"].x1) / 2
        yc = (line["bbox"].y0 + line["bbox"].y1) / 2
        if any(x0 <= xc <= x1 and y0 <= yc <= y1 for x0, y0, x1, y1 in skip):
            continue                                   # inside a figure/caption region
        raw = "".join(s["text"] for s in spans if not is_hidden(s["font"])).strip()
        if re.fullmatch(r"\d{1,4}", raw) and (line["bbox"].y1 < 0.10 * H or line["bbox"].y0 > 0.90 * H):
            continue                                   # running page number
        vis = [s for s in spans if not is_hidden(s["font"]) and s["text"].strip()]
        num_x0 = (vis[-1].get("bbox") or (line["bbox"].x0,))[0] if vis else 0
        if (NUM_RE.match(raw) and len(raw) <= 9 and num_x0 > 0.6 * page.rect.width
                and 0.08 * H <= line["bbox"].y0 and line["bbox"].y1 <= 0.94 * H):
            number = "(" + re.sub(r"[()\s]", "", raw) + ")"       # right-aligned equation number
            hid_raw = "".join(s["text"] for s in spans if is_hidden(s["font"])).strip()
            if hid_raw:                                # its equation sits in the hidden layer on the same line
                marker = add_eq("", "display")
                rec = eqs[-1]
                rec.update({"ocr_hidden": hid_raw, "number": number,
                            "bbox": [round(v, 1) for v in tuple(line["bbox"])]})
                out.append(marker)
                last = None
            else:
                nums.append((line["bbox"], number))
            continue                                   # (any font: Oppenheim sets them in Times)
        if line["is_body"]:
            out.append(render(spans))
            continue
        in_margin = line["bbox"].y0 < 0.08 * H or line["bbox"].y1 > 0.94 * H
        if in_margin:
            continue
        if all(is_hidden(s["font"]) for s in spans):
            hids.append((line["bbox"], "".join(s["text"] for s in spans).strip()))
            continue                                   # hidden-layer-only line: kept only as equation context
        if not any(font_key(s["font"]) == p["family"] for s in spans):
            continue                                   # header, heading, label
        plain = raw
        hid = "".join(s["text"] for s in spans if is_hidden(s["font"])).strip()
        if len(re.findall(r"[A-Za-z]{3,}", plain)) >= p["prose_words"]:
            out.append(render(spans))                  # math-heavy sentence: keep as text
            last = None
            continue
        bb = line["bbox"]
        if (last and out and out[-1] == last[0]
                and bb.y0 - last[1]["_bb"][3] <= 8 and bb.y0 >= last[1]["_bb"][1] - 3):
            rec = last[1]                              # continuation of the same display equation
            rec["ocr_text"] += "\n" + plain
            rec["ocr_hidden"] = (rec.get("ocr_hidden", "") + " " + hid).strip()
            rec["_bb"] = [min(rec["_bb"][0], bb.x0), min(rec["_bb"][1], bb.y0),
                          max(rec["_bb"][2], bb.x1), max(rec["_bb"][3], bb.y1)]
            continue
        marker = add_eq(plain, "display")
        rec = eqs[-1]
        rec["ocr_hidden"] = hid
        rec["_bb"] = [bb.x0, bb.y0, bb.x1, bb.y1]
        last = (marker, rec)
        out.append(marker)

    out.extend(c[1] for c in pending)                  # figures below the last text line
    # finish display equations: bbox for later cropping, equation number if one sits on the same rows
    for rec in (e for e in eqs if e["page"] == page_num and "_bb" in e):
        x0, y0, x1, y1 = rec.pop("_bb")
        rec["bbox"] = [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]
        extra = [t for hb, t in hids if t and x0 - 10 <= (hb.x0 + hb.x1) / 2 <= x1 + 10
                 and y0 - 6 <= (hb.y0 + hb.y1) / 2 <= y1 + 6]
        if extra:
            rec["ocr_hidden"] = (rec.get("ocr_hidden", "") + " " + " ".join(extra)).strip()
        near = [(abs((nb.y0 + nb.y1) / 2 - (y0 + y1) / 2), t) for nb, t in nums
                if y0 - 6 <= (nb.y0 + nb.y1) / 2 <= y1 + 6 and nb.x0 >= x1 - 10]
        if near:
            rec["number"] = min(near)[1]
    return normalize("\n".join(out))


# ---------------------------------------------------------------------------
# Stage 3: chunking
# ---------------------------------------------------------------------------

def chunk_text(pages, out_dir, pages_per_chunk=3, overlap_pages=1):
    d = os.path.join(out_dir, "chunks")
    os.makedirs(d, exist_ok=True)
    manifest, i, k = [], 0, 0
    while i < len(pages):
        win = pages[i:i + pages_per_chunk]
        text = "\n\n".join(f"[PDF page {x['page']}]\n{x['text']}" for x in win)
        with open(os.path.join(d, f"chunk_{k:04d}.txt"), "w") as f:
            f.write(text)
        manifest.append({"chunk_id": k, "path": f"chunks/chunk_{k:04d}.txt",
                         "pages": [x["page"] for x in win], "chars": len(text)})
        k += 1
        i += max(pages_per_chunk - overlap_pages, 1)
    with open(os.path.join(out_dir, "chunk_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(args):
    os.makedirs(args.out, exist_ok=True)
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    doc = fitz.open(args.pdf_path)
    print(f"Loaded {args.pdf_path}: {len(doc)} pages")

    p = detect_body_params(doc)
    if args.body_font:
        p["family"] = args.body_font.lower()[:5]
    if args.min_size:
        p["min"] = args.min_size
    if args.max_size:
        p["max"] = args.max_size
    print(f"[detect] using body family='{p['family']}', size band {p['min']}-{p['max']}pt")

    args.fig_overrides = json.load(open(args.fig_overrides)) if args.fig_overrides else {}
    table_pages = parse_pages(args.table_pages)
    skip = parse_pages(args.skip_pages)
    only = parse_pages(args.only_pages)

    pages, eqs, figs = [], [], []
    raw_chars = kept_chars = 0
    for page in doc:
        n = page.number + 1
        if n in skip or (only and n not in only):
            continue
        printed = (n - args.page_offset) if args.page_offset else None
        raw_chars += len(page.get_text())

        found = find_figures(page, p)
        if n in table_pages and not any(f["kind"] == "table" for f in found):
            W, H = page.rect.width, page.rect.height          # continuation page: the whole text area is table
            area = fitz.Rect(0.10 * W, 0.08 * H, 0.96 * W, 0.94 * H)
            found.append({"kind": "table", "label": None, "rect": area, "caption": "(continued table)",
                          "skip": tuple(area), "cap_boxes": [], "cap_bottom": area.y0})
        caps = []
        if args.captions_in_text:
            for k, f in enumerate(found, 1):
                if f["caption"] and f["caption"] != "(continued table)":
                    fid = f"{'tab' if f['kind'] == 'table' else 'fig'}_p{n}_{k}"
                    caps.append((f["skip"][1], f"<{fid}: {f['caption']}>"))
        text = clean_page(page, eqs, p, skip=[f["skip"] for f in found], captions=caps)
        if args.debug:
            pl = page_lines(page, p)
            print(f"[debug] page {n}: {len(pl)} lines ({sum(l['is_body'] for l in pl)} body), "
                  f"page height {page.rect.height:.0f}, kept {len(text)} chars, {len(found)} caption(s)")
            for f in found:
                print(f"         {f['kind']} skip region={tuple(round(v) for v in f['skip'])} "
                      f"rect={None if f['rect'] is None else tuple(round(v) for v in f['rect'])} "
                      f"caption={f['caption'][:250]!r}")
            if not text:
                print("         (no text kept - first 8 lines below)")
                for l in pl[:8]:
                    print("         ", "BODY" if l["is_body"] else "----", tuple(round(v) for v in l["bbox"]),
                          [(s["font"], round(s["size"], 1), s["text"][:30]) for s in l["spans"]][:3])
        if text:
            kept_chars += len(text)
            pages.append({"page": n, "printed_page": printed, "text": text})

        for k, info in enumerate(found, 1):
            fid = f"{'tab' if info['kind'] == 'table' else 'fig'}_p{n}_{k}"
            rec = {"id": fid, "type": info["kind"], "pdf_page": n, "printed_page": printed,
                   "label": info["label"], "caption": info["caption"], "caption_latex": None, "file": None}
            r = info["rect"]
            if r is not None:
                if args.fig_overrides.get(fid):
                    r = fitz.Rect(args.fig_overrides[fid])
                elif info["kind"] == "figure":
                    r = extend_to_figure_bottom(page, r, info["cap_boxes"], info["cap_bottom"])
                if (r.height < 25 or r.width < 25) and info["kind"] == "figure" and not args.fig_overrides.get(fid):
                    cx0, cx1 = info["col"]                       # retry: everything between header and caption
                    retry = fitz.Rect(cx0, 0.085 * page.rect.height + 2, cx1, info["cap_top"] - 2)
                    if retry.height >= 25:
                        print(f"[note] page {n} {fid}: crop retried from the header margin (check it)")
                        rec["note"] = "crop top set from header margin; check manually"
                        r = extend_to_figure_bottom(page, retry, info["cap_boxes"], info["cap_bottom"])
                if r.height < 25 or r.width < 25:
                    print(f"[warn] page {n} {fid}: crop region too small {tuple(round(v) for v in r)}"
                          f" - skipped (caption: {info['caption'][:60]!r})")
                    rec["note"] = "crop region empty; check this page manually"
                else:
                    try:
                        r2 = save_figure(doc, page, r, os.path.join(fig_dir, fid), trim=not args.no_trim,
                                         mask=info["cap_boxes"] if info["kind"] == "figure" else ())
                        rec["file"] = f"figures/{fid}.pdf"
                        rec["png"] = f"figures/{fid}.png"
                        rec["bbox"] = [round(r2.x0, 1), round(r2.y0, 1), round(r2.x1, 1), round(r2.y1, 1)]
                    except Exception as e:
                        print(f"[warn] page {n} {fid}: crop failed ({e})")
                        rec["note"] = f"crop failed: {e}"
            figs.append(rec)

    with open(os.path.join(args.out, "clean_text.json"), "w") as f:
        json.dump(pages, f, indent=2)
    with open(os.path.join(args.out, "clean_text.txt"), "w") as f:
        f.write("\n\n".join(f"[PDF page {x['page']}]\n{x['text']}" for x in pages))
    with open(os.path.join(args.out, "equations.json"), "w") as f:
        json.dump(eqs, f, indent=2)
    with open(os.path.join(args.out, "captions.json"), "w") as f:
        json.dump(figs, f, indent=2)
    manifest = chunk_text(pages, args.out, args.pages_per_chunk)

    kinds = Counter(e["kind"] for e in eqs)
    print(f"[text]  {len(pages)} pages, {kept_chars} chars kept of {raw_chars} raw")
    print(f"[eq]    {len(eqs)} markers ({dict(kinds)})")
    print(f"[fig]   {sum(1 for f in figs if f['file'] and f['type'] == 'figure')} figures and "
          f"{sum(1 for f in figs if f['file'] and f['type'] == 'table')} tables cropped")
    print(f"[chunk] {len(manifest)} chunks -> {args.out}/chunks/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clean a scanned Signals textbook for KG extraction.")
    ap.add_argument("pdf_path")
    ap.add_argument("--out", default="output")
    ap.add_argument("--skip-pages", nargs="*", default=[],
                    help="PDF page numbers (1-based) or ranges, e.g. 1-31 960-987")
    ap.add_argument("--only-pages", nargs="*", default=[],
                    help="process only these pages/ranges (for testing), e.g. 40-42")
    ap.add_argument("--page-offset", type=int, default=0,
                    help="PDF page minus printed page (31 for your Oppenheim copy)")
    ap.add_argument("--body-font", default=None,
                    help="override detected body font family prefix, e.g. Times")
    ap.add_argument("--min-size", type=float, default=None)
    ap.add_argument("--max-size", type=float, default=None)
    ap.add_argument("--pages-per-chunk", type=int, default=3)
    ap.add_argument("--table-pages", nargs="*", default=[],
                    help="continuation pages that are all table (no TABLE title on them), e.g. 360 422")
    ap.add_argument("--fig-overrides", default=None,
                    help='JSON file {"fig_p62_2": [x0, y0, x1, y1]} to hand-fix individual crops')
    ap.add_argument("--no-captions-in-text", dest="captions_in_text", action="store_false",
                    help="keep captions out of the text (default: captions stay in the text as <fig_pN_k: ...>)")
    ap.add_argument("--no-trim", action="store_true", help="do not trim figure crops to the ink")
    ap.add_argument("--debug", action="store_true", help="print per-page diagnostics")
    run(ap.parse_args())