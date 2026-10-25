"""Add a generated "Related" section to the end of each note in public/.
Notes listed in RELATED get hand-picked links instead.

Relatedness = TF-IDF cosine over title / summary / tags / H1-H2 headings,
plus a bonus for shared frontmatter tags and shared topics in build_index.TOPICS.
The section sits between <!-- related --> markers and is replaced on every run.

  python3 index/build_related.py                  # preview all notes
  python3 index/build_related.py 2017/01-12_git   # preview matching notes
  python3 index/build_related.py --write          # update the notes
"""
import collections, math, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_index as bi

PUB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")
bi.PUB = PUB  # build_index reads sys.argv[1] as the repo path
MAX_LINKS, MIN_LINKS = 5, 2
MIN_SCORE, REL_SCORE = 0.25, 0.5  # keep links scoring >= max(MIN_SCORE, REL_SCORE * best)
WEAK = 0.4  # notes whose best match is below this get only MIN_LINKS links
TAG_BONUS, TOPIC_BONUS = 0.15, 0.1
STOP = set("the and for with of to in on a an by from use using how notes note basic basics advance advanced".split())
BLOCK = re.compile(r"\s*<!-- related -->.*?<!-- /related -->\s*$", re.S)

# Hand-picked links, note prefix -> related note prefixes (as in build_index.TOPICS).
# They replace the computed links; an empty list means no Related section.
RELATED = {
    "2026/09-16_taiwan": ["2021/11-14_shandong", "2024/10-08_ielts", "2026/02-26_gym"],
    "Dainty_factory/": ["2026/02-26_gym"],
    "2018/09-20_JP": ["2024/10-08_ielts"],
    "2024/10-08_ielts": ["2018/09-20_JP", "2026/09-16_taiwan"],
    "2021/11-14_shandong": ["2026/09-16_taiwan"],
    "2026/02-26_gym": ["Dainty_factory/", "2026/09-16_taiwan"],
}


def tokens(s):
    s = s.lower()
    out = re.findall(r"[a-z][a-z0-9+#]+", s)
    for run in re.findall(r"[一-鿿]+", s):  # CJK: character bigrams
        out += [run[i:i + 2] for i in range(len(run) - 1)]
    return [w for w in out if w not in STOP]


def load(rel):
    raw = open(os.path.join(PUB, rel), encoding="utf-8").read()
    raw = BLOCK.sub("", raw)
    meta = bi.frontmatter(os.path.join(PUB, rel))
    tags = set(t.strip() for t in meta.get("tags", "").strip("[]").split(",") if t.strip())
    heads = " ".join(re.findall(r"^#{1,2} (.+)$", re.sub(r"```.*?```", "", raw, flags=re.S), re.M))
    stem = re.sub(r"^\d\d-\d\d?_", "", os.path.basename(rel)[:-3]).replace("_", " ")
    words = (tokens(stem) * 3 + tokens(meta.get("summary", "")) * 2
             + tokens(" ".join(tags).replace("-", " ")) * 2 + tokens(heads))
    return tags, collections.Counter(words)


def scores(notes):
    topic = collections.defaultdict(set)
    for key, _, _, groups, _ in bi.TOPICS:
        for _, prefixes in groups:
            for n in notes:
                if any(n.startswith(p) for p in prefixes):
                    topic[n].add(key)
    tags, docs = zip(*(load(n) for n in notes))
    df = collections.Counter(w for d in docs for w in d)
    vecs = []
    for d in docs:
        v = {w: (1 + math.log(c)) * math.log(len(notes) / df[w]) for w, c in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1
        vecs.append({w: x / norm for w, x in v.items()})
    S = {}
    for i, a in enumerate(notes):
        for j, b in enumerate(notes):
            if i < j:
                small, big = sorted((vecs[i], vecs[j]), key=len)
                s = sum(x * big.get(w, 0) for w, x in small.items())
                s += TAG_BONUS * len(tags[i] & tags[j]) + TOPIC_BONUS * bool(topic[a] & topic[b])
                S[a, b] = S[b, a] = s
    return S


def pick(n, notes, S):
    ranked = sorted((S[n, m], m) for m in notes if m != n)[::-1]
    if ranked[0][0] < WEAK:
        return [m for _, m in ranked[:MIN_LINKS]]
    cut = max(MIN_SCORE, REL_SCORE * ranked[0][0])
    keep = [m for s, m in ranked[:MAX_LINKS] if s >= cut]
    return keep if len(keep) >= MIN_LINKS else [m for _, m in ranked[:MIN_LINKS]]


def link(src, dst, info):
    url = os.path.relpath(dst, os.path.dirname(src)).replace(" ", "%20").replace("(", "%28").replace(")", "%29")
    return f"- [{bi.esc(info['title'], 80)}]({url})"


def resolve(prefix, notes):
    found = [m for m in notes if m.startswith(prefix)]
    if len(found) != 1:
        sys.exit(f"RELATED prefix {prefix!r} matches {len(found)} notes")
    return found[0]


def manual(n, notes):
    for prefix, related in RELATED.items():
        if n.startswith(prefix):
            return [resolve(p, notes) for p in related]
    return None


def block(n, related):
    if not related:
        return ""
    lines = ["<!-- related -->", "# Related"] + [link(n, m, bi.note_info(m)) for m in related]
    return "\n".join(lines + ["***", "<!-- /related -->"])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write" in sys.argv
    notes = sorted(
        os.path.relpath(os.path.join(d, f), PUB)
        for d, _, fs in os.walk(PUB) for f in fs if f.endswith(".md") and f != "README.md"
    )
    S = scores(notes)
    for n in notes:
        if args and not any(a in n for a in args):
            continue
        related = manual(n, notes)
        text = block(n, pick(n, notes, S) if related is None else related)
        if not write:
            print(f"== {n}\n{text}\n")
            continue
        path = os.path.join(PUB, n)
        raw = open(path, "rb").read().decode("utf-8")
        nl = "\r\n" if "\r\n" in raw else "\n"
        body = BLOCK.sub("", raw.replace("\r\n", "\n")).rstrip("\n")
        new = (body + "\n\n" + text + "\n" if text else body + "\n").replace("\n", nl)
        if new != raw:
            open(path, "wb").write(new.encode("utf-8"))
    if write:
        print(f"Related sections written for {len(notes)} notes")


if __name__ == "__main__":
    main()
