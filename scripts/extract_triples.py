"""
Pilot triple extraction + relation-frequency table.

Step in the pipeline: clean chunks + ontology -> triples -> how often each relation occurs
-> cut the long tail (keep the ~40 most frequent relation types).

Usage (from the project root, after clean_pipeline_pymupdf.py has made out/chunks):
  # 1. cost check, no API calls
  python extract_triples.py --chunks out --ontology ontology/signals_ontology.json --dry-run
  # 2. pilot run on 40 evenly spaced chunks
  python extract_triples.py --chunks out --provider anthropic --model <model-id> --n-chunks 40
  python extract_triples.py --chunks out --provider google    --model <model-id> --n-chunks 40
  # 3. re-make the table from saved triples (change cutoffs without paying again)
  python extract_triples.py --analyze-only --top-n 40 --min-chunk-share 0.02
  # offline test
  python extract_triples.py --chunks out --provider mock --n-chunks 10

Keys: ANTHROPIC_API_KEY / GOOGLE_API_KEY (or GEMINI_API_KEY) in the environment.
Outputs (in --pilot-dir, default pilot/): raw/chunk_XXXX.json (cache; reruns skip these),
triples.jsonl, relation_frequency.csv, relation_frequency.json, other_relations.csv
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

PROMPT = """You are building a knowledge graph from a Signals & Systems textbook (Oppenheim & Willsky).

Extract knowledge triples (head, relation, tail) from the passage below.

Rules:
- head and tail are short noun phrases using the textbook's own terms (e.g. "unit impulse", "LTI system", "region of convergence").
{class_rule}{relation_rule}
- Only extract facts the passage states. No outside knowledge. No section numbers or exercise text.
- evidence is a short VERBATIM quote (<= 200 chars) from the passage that supports the triple.
- If an equation marker such as <eq_...> sits next to the fact, put its id in equation_ids.
- Figure and table captions appear in the passage as <fig_...: caption text> or <tab_...: caption text>. Facts stated
  in a caption may be extracted; put that marker's id (e.g. fig_p62_1) in figure_ids.
- Prefer fewer, correct triples to many weak ones.
- Skip tautologies (e.g. "time-invariant system has_property time invariance") and vacuous tails
  (e.g. "Fourier series coefficients"): the tail must add information beyond restating the head.
- Prefer general facts over problem-specific names (e.g. e[n], p[n]) unless the example illustrates a general principle.
- Use has_transfer_function when the thing given is a system function H(s) or H(z); use governed_by_equation
  only for differential or difference equations.
{other_pref}

{class_block}RELATIONS:
{relations}

Return ONLY a JSON list, no prose:
[{{"head":"...","relation":"...","tail":"...",{type_fields}"evidence":"...","equation_ids":[],"figure_ids":[]}}]

PASSAGE:
\"\"\"
{passage}
\"\"\"
"""


def load_ontology(path):
    o = json.load(open(path))
    o.setdefault("classes", [])  # classes are optional: the base relation list has none
    defs = {r["name"]: r.get("definition", "") for r in o["relations"]}
    return [c["name"] for c in o["classes"]], [r["name"] for r in o["relations"]], defs


def load_chunks(root):
    man = json.load(open(os.path.join(root, "chunk_manifest.json")))
    for m in man:
        m["text"] = open(os.path.join(root, m["path"])).read()
    return man


def sample_chunks(chunks, n, seed=0):
    """Evenly spaced across the book (stratified), so every chapter is represented."""
    if n >= len(chunks):
        return chunks
    step = len(chunks) / n
    rng = random.Random(seed)
    return [chunks[min(int(i * step + rng.random() * step), len(chunks) - 1)] for i in range(n)]


def build_prompt(chunk, classes, relations, defs=None, strict=False):
    """strict=True: seed-KG mode. Only the listed relations are allowed; OTHER is not offered."""
    if strict:
        rule = ("- relation must be exactly one of the RELATIONS. If no relation fits the fact, "
                "SKIP the fact. Never invent a relation.")
        pref = ("- Use the relation whose definition best matches the fact (e.g. affects_stability_of for "
                "stability determination, is_subclass_of for \"is a\").")
    else:
        rule = ('- relation must be one of the RELATIONS. If none fits, use "OTHER:<snake_case_name>" '
                '- do NOT force a bad fit.')
        pref = ("- Prefer a listed relation over OTHER whenever one fits (e.g. affects_stability_of for "
                "stability determination,\n  is_subclass_of for \"is a\"). If you must use OTHER, make it a "
                "short, general snake_case verb phrase.")
    if defs and any(defs.values()):
        rel_text = "\n".join(f"- {r}: {defs[r]}" for r in relations)
    else:
        rel_text = ", ".join(relations)
    if classes:
        class_rule = "- head_type and tail_type must each be one of the CLASSES.\n"
        class_block = "CLASSES:\n" + ", ".join(classes) + "\n\n"
        type_fields = '"head_type":"...","tail_type":"...",'
    else:
        class_rule = class_block = type_fields = ""
    return PROMPT.format(class_rule=class_rule, class_block=class_block, type_fields=type_fields,
                         relations=rel_text, passage=chunk["text"], relation_rule=rule, other_pref=pref)


def clean_triples(triples, relations, strict):
    """In strict mode drop any triple whose relation is not in the ontology (the model may still slip)."""
    if not strict:
        return triples
    ok = set(relations)
    return [t for t in triples if t["relation"] in ok]


def parse_triples(raw):
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for t in data:
        if isinstance(t, dict) and t.get("head") and t.get("tail") and t.get("relation"):
            out.append({k: t.get(k, [] if k in ("equation_ids", "figure_ids") else "") for k in
                        ("head", "head_type", "relation", "tail", "tail_type", "evidence", "equation_ids",
                         "figure_ids")})
    return out


# ---------------------------------------------------------------- providers
def call_anthropic(prompt, model, max_tokens=4096):
    import anthropic
    c = anthropic.Anthropic()
    r = c.messages.create(model=model, max_tokens=max_tokens, temperature=0,
                          messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in r.content if b.type == "text"), r.usage.input_tokens, r.usage.output_tokens


_GOOGLE_CLIENT = None


def call_google(prompt, model, max_tokens=16384):  # thinking tokens count toward this limit
    global _GOOGLE_CLIENT
    from google import genai
    if _GOOGLE_CLIENT is None:  # one long-lived client; a throwaway one gets closed mid-request
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        _GOOGLE_CLIENT = genai.Client(api_key=key)
    c = _GOOGLE_CLIENT
    r = c.models.generate_content(model=model, contents=prompt,
                                  config={"temperature": 0, "max_output_tokens": max_tokens})
    u = r.usage_metadata
    return r.text or "", getattr(u, "prompt_token_count", 0), getattr(u, "candidates_token_count", 0)


def call_mock(prompt, model, max_tokens=0):
    """Deterministic fake so the whole pipeline can be tested offline."""
    block = re.search(r"RELATIONS:\n(.*?)\n\n", prompt, re.S).group(1)
    rels = re.findall(r"^- (\w+):", block, re.M) or block.split(", ")
    h = sum(map(ord, prompt[-300:]))
    rng = random.Random(h)
    out = []
    for i in range(rng.randint(3, 8)):
        r = rng.choice(rels[:12] if rng.random() < .7 else rels)
        if rng.random() < .1 and "SKIP the fact" not in prompt:
            r = "OTHER:" + rng.choice(["is_approximated_by", "has_unit", "is_illustrated_by"])
        out.append({"head": f"entity_{rng.randint(1,30)}", "head_type": "Signal", "relation": r,
                    "tail": f"entity_{rng.randint(1,30)}", "tail_type": "System",
                    "evidence": "mock", "equation_ids": []})
    return json.dumps(out), len(prompt) // 4, 200


PROVIDERS = {"anthropic": call_anthropic, "google": call_google, "mock": call_mock}


# ---------------------------------------------------------------- analysis
def analyze(pilot_dir, relations, top_n, min_chunk_share, merge=None):
    path = os.path.join(pilot_dir, "triples.jsonl")
    triples = [json.loads(l) for l in open(path)]
    if not triples:
        print("\nNo triples were extracted (every call failed or returned nothing). "
              "Fix the errors above and rerun; nothing to tabulate.")
        return
    n_chunks = len({t["chunk_id"] for t in triples}) or 1
    # denominator = requested chunks that actually have results (incl. ones with zero triples);
    # stale cache files from other runs are NOT counted
    pc_path = os.path.join(pilot_dir, "processed_chunks.json")
    if os.path.exists(pc_path):
        n_chunks = max(1, len(json.load(open(pc_path))["chunk_ids"]))
    cnt, chunks, ex = Counter(), defaultdict(set), {}
    includes = defaultdict(Counter)  # canonical -> {merged-in name: count}
    for t in triples:
        r0 = t["relation"]
        r = (merge or {}).get(r0, r0)
        if r != r0:
            includes[r][r0] += 1
        cnt[r] += 1
        chunks[r].add(t["chunk_id"])
        ex.setdefault(r, f'{t["head"]} -> {t["tail"]}')
    total = sum(cnt.values()) or 1
    rows, cum = [], 0
    for rank, (r, n) in enumerate(cnt.most_common(), 1):
        cum += n
        in_onto = r in relations
        pc = len(chunks[r]) / n_chunks
        keep = in_onto and rank <= top_n and pc >= min_chunk_share
        rows.append({"rank": rank, "relation": r, "n_triples": n, "pct_triples": round(100 * n / total, 2),
                     "cum_pct_triples": round(100 * cum / total, 2), "n_chunks": len(chunks[r]),
                     "pct_chunks": round(100 * pc, 2), "in_ontology": in_onto,
                     "keep": keep, "includes": "; ".join(f"{k}:{v}" for k, v in includes[r].items()),
                     "example": ex[r]})
    unused = [r for r in relations if r not in cnt]
    with open(os.path.join(pilot_dir, "relation_frequency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["relation"])
        w.writeheader()
        w.writerows(rows)
    json.dump({"n_triples": total, "n_chunks": n_chunks, "top_n": top_n,
               "min_chunk_share": min_chunk_share, "rows": rows, "unused_ontology_relations": unused},
              open(os.path.join(pilot_dir, "relation_frequency.json"), "w"), indent=2)
    other = [r for r in rows if not r["in_ontology"]]
    with open(os.path.join(pilot_dir, "other_relations.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["relation", "n_triples", "n_chunks", "example"])
        for r in other:
            w.writerow([r["relation"], r["n_triples"], r["n_chunks"], r["example"]])
    kept = [r for r in rows if r["keep"]]
    print(f"\n{total} triples from {n_chunks} chunks; {len(rows)} distinct relations "
          f"({len(other)} OTHER:*), {len(unused)} ontology relations never used")
    print(f"{'rank':>4}  {'relation':36} {'triples':>7} {'%trip':>6} {'cum%':>6} {'%chunks':>8}  keep")
    for r in rows[:70]:
        print(f"{r['rank']:>4}  {r['relation'][:36]:36} {r['n_triples']:>7} {r['pct_triples']:>6} "
              f"{r['cum_pct_triples']:>6} {r['pct_chunks']:>8}  {'*' if r['keep'] else ''}"
              + (f"   (+ {r['includes']})" if r["includes"] else ""))
    print(f"\nKEEP {len(kept)} relations covering "
          f"{round(sum(r['pct_triples'] for r in kept), 1)}% of triples "
          f"(top_n={top_n}, min_chunk_share={min_chunk_share})")
    print("Unused:", ", ".join(unused) or "-")
    print(f"Wrote {pilot_dir}/relation_frequency.csv | .json | other_relations.csv")


# ---------------------------------------------------------------- main
def load_merge(path):
    if not path:
        return None
    m = json.load(open(path))
    return m.get("merge", m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", default="out", help="dir holding chunk_manifest.json and chunks/")
    ap.add_argument("--ontology", default="ontology/signals_relations_base.json")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="mock")
    ap.add_argument("--model", default="")
    ap.add_argument("--n-chunks", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pilot-dir", default="pilot")
    ap.add_argument("--dry-run", action="store_true", help="estimate cost, call nothing")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--no-other", action="store_true",
                    help="seed-KG mode: only the ontology's relations are allowed; OTHER:* is not offered and is dropped")
    ap.add_argument("--show", type=int, default=0,
                    help="print N random saved triples with evidence for manual review, then exit")
    ap.add_argument("--merge-map", default="",
                    help="JSON {\"merge\": {old_relation: canonical_relation}} - counts near-duplicates together")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds to wait between requests (helps with free-tier per-minute limits)")
    ap.add_argument("--top-n", type=int, default=40, help="max relation types to keep")
    ap.add_argument("--min-chunk-share", type=float, default=0.0,
                    help="relation must appear in at least this fraction of chunks (0.02 = 2%%)")
    ap.add_argument("--price-in", type=float, default=3.0, help="$ per 1M input tokens (set to your model's)")
    ap.add_argument("--price-out", type=float, default=15.0, help="$ per 1M output tokens")
    a = ap.parse_args()

    classes, relations, defs = load_ontology(a.ontology)
    os.makedirs(os.path.join(a.pilot_dir, "raw"), exist_ok=True)
    if a.show:
        ts = [json.loads(l) for l in open(os.path.join(a.pilot_dir, "triples.jsonl"))]
        random.Random(1).shuffle(ts)
        for k, t in enumerate(ts[:a.show], 1):
            ht = f"({t['head_type']}) " if t.get("head_type") else ""
            tt = f"({t['tail_type']}) " if t.get("tail_type") else ""
            print(f"[{k}] {ht}{t['head']}  --{t['relation']}-->  {tt}{t['tail']}")
            print(f"     \"{t['evidence']}\"   [chunk {t['chunk_id']}, pdf p.{t['pages'][0]}-{t['pages'][-1]}]\n")
        return
    if a.analyze_only:
        return analyze(a.pilot_dir, relations, a.top_n, a.min_chunk_share, load_merge(a.merge_map))

    chunks = sample_chunks(load_chunks(a.chunks), a.n_chunks, a.seed)
    prompts = [build_prompt(c, classes, relations, defs, a.no_other) for c in chunks]
    tin = sum(len(p) for p in prompts) // 4
    tout = len(chunks) * 900  # ~15-25 triples of JSON per chunk
    cost = tin / 1e6 * a.price_in + tout / 1e6 * a.price_out
    print(f"{len(chunks)} chunks | ~{tin:,} input + ~{tout:,} output tokens | "
          f"~${cost:.2f} at ${a.price_in}/${a.price_out} per 1M (rough; check your model's prices)")
    if a.dry_run:
        return
    call = PROVIDERS[a.provider]
    seen = tot_in = tot_out = fails = 0
    for c, p in zip(chunks, prompts):
        rp = os.path.join(a.pilot_dir, "raw", f"chunk_{c['chunk_id']:04d}.json")
        if os.path.exists(rp):
            continue
        raw = None
        for attempt in range(6):  # retry temporary overload/rate errors with growing waits
            try:
                raw, i, o = call(p, a.model)
                break
            except Exception as e:
                msg = str(e)
                temp = any(s in msg for s in ("503", "UNAVAILABLE", "overloaded", "high demand",
                                              "429", "RESOURCE_EXHAUSTED", "500", "504", "timed out",
                                              "nodename nor servname", "Temporary failure",
                                              "Connection", "Errno 8"))
                if not temp or "limit: 0" in msg or attempt == 5:
                    print(f"chunk {c['chunk_id']} failed: {msg[:300]}", file=sys.stderr)
                    break
                wait = min(10 * 2 ** attempt, 120)
                print(f"chunk {c['chunk_id']}: busy, retrying in {wait}s (try {attempt + 1}/5)",
                      file=sys.stderr)
                time.sleep(wait)
        if raw is None:
            fails += 1
            if fails >= 3:
                print("\nStopping: 3 chunks in a row failed. Check your internet connection / quota, "
                      "then rerun the same command; finished chunks are cached.", file=sys.stderr)
                break
            continue
        fails = 0
        if a.delay:
            time.sleep(a.delay)
        tot_in += i
        tot_out += o
        json.dump({"chunk_id": c["chunk_id"], "pages": c["pages"], "raw": raw,
                   "triples": clean_triples(parse_triples(raw), relations, a.no_other)}, open(rp, "w"))
        seen += 1
        print(f"chunk {c['chunk_id']:04d} done ({seen}/{len(chunks)})", end="\r")
    print(f"\nAPI tokens used this run: {tot_in:,} in / {tot_out:,} out "
          f"= ~${tot_in/1e6*a.price_in + tot_out/1e6*a.price_out:.2f}")
    wanted = {c["chunk_id"] for c in chunks}
    have = sorted(int(fn[6:10]) for fn in os.listdir(os.path.join(a.pilot_dir, "raw"))
                  if fn.endswith(".json") and int(fn[6:10]) in wanted)
    json.dump({"chunk_ids": have, "requested": len(wanted)},
              open(os.path.join(a.pilot_dir, "processed_chunks.json"), "w"))
    print(f"{len(have)} of {len(wanted)} requested chunks have results"
          + ("" if len(have) == len(wanted) else
             f" - {len(wanted) - len(have)} missing; rerun the same command to fill them in"))
    with open(os.path.join(a.pilot_dir, "triples.jsonl"), "w") as f:
        for fn in sorted(os.listdir(os.path.join(a.pilot_dir, "raw"))):
            d = json.load(open(os.path.join(a.pilot_dir, "raw", fn)))
            if d["chunk_id"] not in wanted:
                continue
            for t in d["triples"]:
                f.write(json.dumps({**t, "chunk_id": d["chunk_id"], "pages": d["pages"]}) + "\n")
    analyze(a.pilot_dir, relations, a.top_n, a.min_chunk_share, load_merge(a.merge_map))


if __name__ == "__main__":
    main()