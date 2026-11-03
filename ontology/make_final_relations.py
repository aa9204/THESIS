"""
Builds the FINAL relation list for the seed-KG run from the base run's frequency table.

  python ontology/make_final_relations.py --freq base_run/relation_frequency.json \
      --base ontology/signals_relations_base.json --merge-map ontology/merge_map.json \
      --out ontology/signals_relations_final.json

Keeps the relations flagged keep=true in the frequency table (rerun the table with --merge-map and
--top-n 40 first). Merged relations get the broadened definition from merge_map.json["definitions"].
Edit the output by hand if the cut falls in a tie (the script prints a warning).
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--freq", default="base_run/relation_frequency.json")
ap.add_argument("--base", default="ontology/signals_relations_base.json")
ap.add_argument("--merge-map", default="ontology/merge_map.json")
ap.add_argument("--out", default="ontology/signals_relations_final.json")
ap.add_argument("--add", nargs="*", default=[], help="relations to add by hand (e.g. to break a tie)")
ap.add_argument("--drop", nargs="*", default=[], help="relations to remove from the cut")
a = ap.parse_args()

freq = json.load(open(a.freq))
base = {r["name"]: r.get("definition", "") for r in json.load(open(a.base))["relations"]}
mm = json.load(open(a.merge_map))
defs = {**base, **mm.get("definitions", {})}

rows_by_name = {r["relation"]: r for r in freq["rows"]}
for name in a.add + a.drop:
    if name not in rows_by_name:
        raise SystemExit(f"unknown relation in --add/--drop: {name}")
keep = [r for r in freq["rows"] if (r["keep"] and r["relation"] not in a.drop) or r["relation"] in a.add]
keep.sort(key=lambda r: r["rank"])
out = {"version": "final", "relations": [
    {"name": r["relation"], "definition": defs.get(r["relation"], ""), "n_triples_base_run": r["n_triples"],
     "includes": r.get("includes", "")} for r in keep]}
json.dump(out, open(a.out, "w"), indent=2)
kept_names = {r["relation"] for r in keep}
last = min(r["n_triples"] for r in keep)
tied = [r["relation"] for r in freq["rows"] if r["n_triples"] == last and r["relation"] not in kept_names]
print(f"wrote {len(keep)} relations to {a.out}")
if tied:
    print(f"NOTE: {len(tied)} other relations are tied at {last} triples and left out: {tied}")
missing = [r["name"] for r in out["relations"] if not r["definition"]]
if missing:
    print("no definition for:", missing)