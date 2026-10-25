"""Generate index/*.md (Maps of Content) for Atom_notebook/public.

Run from anywhere: python3 index/build_index.py
Section links use GitHub's anchor when Obsidian resolves the same anchor,
otherwise Obsidian's (heading text); on GitHub those open the note at the top.
"""
import os, re, sys, unicodedata

REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUB = os.path.join(REPO, "public")
OUT = os.path.join(REPO, "index")

TOPICS = [
    ("embedded_arm", "Embedded Linux & ARM",
     "ARM SoC basics, hardware interfaces, Linux driver development, bootloader / kernel porting, NPU boards.",
     [("", ["2013_ARM_embedded/", "2013_Basic/05-16_", "2021/01-22_rknn"])],
     ["model_deployment"]),
    ("c_linux_system", "C/C++ & Linux System Programming",
     "C language, compilation toolchain, Linux processes / threads / IPC / sockets, Qt and C++.",
     [("", ["2013_Basic/05-17_", "2013_Basic/05-24_", "2013_Basic/05-31_", "2013_Basic/06-03_",
            "2013_Linux_Advance/", "2016_QT_c++/", "2023/07-06_opencv_cpp", "2024/02-26_clang"])],
     ["embedded_arm"]),
    ("shell_tools", "Shell, Linux & Dev Tools",
     "Ubuntu / macOS usage, shell scripting and text processing, Vim, git / GitHub, editors, Markdown.",
     [("", ["2013_Basic/05-12_", "2017/01-04_", "2017/01-17_", "2017/01-12_", "2017/01-14_",
            "2017/05-25_", "2017/07-07_", "2026/10-14_macos", "2026/10-25_obsidian"])],
     ["c_linux_system"]),
    ("other_languages", "Other Languages",
     "Java and Android, Go (incl. cgo), R, TypeScript / Node, Flutter.",
     [("Java & Android", ["2013_Java_Android/"]),
      ("Go", ["2019/12-24_go", "2020/01-06_go", "2020/02-11_cgo"]),
      ("R, TypeScript, Flutter", ["2017/07-30_R", "2026/10-10_typescript", "2026/03-09_flutter"])],
     ["python_data"]),
    ("python_data", "Python, Data & Backend",
     "Python language and modules, numpy / pandas / matplotlib, Spark, SQL, Neo4j, Flask, Kubernetes.",
     [("Python", ["2017/02-06_", "2018/03-16_"]),
      ("Data analysis & visualization", ["2017/03-09_", "2017/04-09_", "2018/06-26_", "2017/05-09_"]),
      ("Databases & services", ["2017/08-06_sql", "2018/07-09_neo4j", "2026/03-20_neo4j",
                                "2018/08-01_flask", "2018/05-16_kubernetes"])],
     ["classic_ml"]),
    ("classic_ml", "Classic ML, Recommenders & NLP",
     "Supervised / unsupervised ML from scratch, scikit-learn, recommender systems, early chatbot / NLP work.",
     [("", ["2017/05-30_", "2017/06-28_", "2017/07-25_", "2018/08-13_surprise", "2018/07-04_course",
            "2018/10-24_chatbot", "2022/08-03_decomposition"])],
     ["python_data", "dl_frameworks"]),
    ("dl_frameworks", "Deep Learning Frameworks",
     "TensorFlow 1.x / 2.x and Keras, PyTorch internals, JAX Haiku, MindSpore, Caffe, HF transformers.",
     [("TensorFlow & Keras", ["2018/04-19_", "2018/08-27_", "2018/09-06_", "2019/10-22_",
                              "2021/01-25_", "2023/01-19_keras_for_torch"]),
      ("PyTorch & others", ["2024/07-19_torch_fx", "2023/12-24_torch_dump", "2024/09-30_transformers",
                            "2021/08-13_haiku", "2022/07-20_mindspore", "2019/08-12_caffe"])],
     ["backbones_imagenet", "model_deployment"]),
    ("face_recognition", "Face Recognition & Anti-Spoofing",
     "Face detection, InsightFace training in Keras, face quality, presentation-attack detection.",
     [("Recognition & training", ["2018/04-16_face", "2019/11-18_keras_insightface",
                                  "2020/12-01_Insightface", "2021/04-07_insightface", "2022/01-10_face_quality"]),
      ("Detection", ["2020/08-21_face_detect", "2021/11-05_yolov5face"]),
      ("Anti-spoofing", ["2019/08-20_anti", "2020/04-10_SiW"])],
     ["backbones_imagenet", "computer_vision"]),
    ("computer_vision", "Computer Vision",
     "Image processing, detection (EfficientDet / YOLO), segmentation (UNet / SAM), CLIP and diffusion.",
     [("Image processing", ["2018/10-11_image", "2020/05-02_computer", "2024/03-16_defog"]),
      ("Detection & datasets", ["2022/01-11_efficientdet", "2022/02-17_yolox", "2022/11-26_yolov7",
                                "2023/04-03_yolov8", "2022/10-21_custom", "2022/08-06_GTSRB", "2019/09-27_car"]),
      ("Segmentation", ["2019/09-05_physic", "2023/10-16_sam"]),
      ("Multimodal & generative", ["2023/05-08_clip", "2023/08-27_diffusion", "2021/02-25_ML"])],
     ["face_recognition", "backbones_imagenet"]),
    ("backbones_imagenet", "Backbones & ImageNet Training",
     "Porting PyTorch models to Keras (keras_cv_attention_models), ImageNet training recipes, new architectures.",
     [("", ["2021/08-20_pytorch_to_keras", "2021/10-14_imagenet", "2022/01-09_imagenet",
            "2022/05-18_advance", "2023/04-25_eva02", "2023/04-29_deformable", "2023/07-09_deepmad",
            "2024/09-2_mamba"])],
     ["model_deployment", "dl_frameworks"]),
    ("model_deployment", "Model Conversion, Compression & Inference",
     "MMdnn / ONNX / TFLite conversion, TensorRT, quantization and pruning, benchmarks.",
     [("", ["2019/07-26_model", "2023/12-12_onnx", "2019/08-19_tensorrt", "2023/05-27_kecam",
            "2023/09-14_tflite", "2019/09-19_model_pruning", "2022/10-18_quant"])],
     ["embedded_arm", "backbones_imagenet"]),
    ("llm_agents", "LLMs & AI Agents",
     "GPT / Llama training, LLM serving (vLLM, llama.cpp), agent frameworks and coding agents.",
     [("Models & training", ["2023/02-27_gpt", "2023/08-06_llama2", "2024/03-23_llm"]),
      ("Serving", ["2025/07-05_vllm", "2026/02-01_llama", "2026/03-11_server"]),
      ("Agents & tools", ["2026/02-02_gemini", "2026/02-28_claude", "2026/07-27_", "2026/01-29_openclaw",
                          "2026/05-08_", "2026/05-10_agno",
                          "2026/08-20_deepseek"])],
     ["dl_frameworks", "nus_courses"]),
    ("nus_courses", "NUS MSc Coursework",
     "Course notes from the NUS MSc in AI & Innovation (2025–2026).",
     [("Technical", ["2025/08-11_nus", "2026/01-16_", "2026/01-17_", "2026/08-11_it5005",
                     "2026/08-17_eca5103", "2026/08-17_is3109", "2026/08-14_eca5307"]),
      ("Innovation, business & society", ["2025/08-12_nus", "2025/08-13_nus", "2025/08-14_nus",
                                          "2026/01-14_", "2026/01-15_nmc", "2026/08-14_nmc", "2026/08-13_ace",
                                          "2026/08-10_"]),
      ("Law, governance & ethics", ["2026/01-13_", "2026/01-15_ll", "2026/08-11_ll"])],
     ["llm_agents"]),
    ("life_languages", "Life & Languages",
     "Japanese and IELTS, fitness, recipes, travel.",
     [("", ["2018/09-20_JP", "2024/10-08_ielts", "2026/02-26_gym", "Dainty_factory/",
            "2026/09-16_taiwan", "2021/11-14_shandong"])],
     []),
]

SKIP_SECTIONS = {"目录", "table of contents", "toc", "links", "链接", "q / a", "q and a", "qa", "info", "related"}
MAX_SECTIONS = 8

FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# Obsidian's stripHeading(): ASCII punctuation -> space, collapse spaces; matched case-insensitively
OBSIDIAN_STRIP = re.compile(r'[!"#$%&()*+,.:;<=>?@^`{|}~\\/\[\]\r\n]')


def obsidian_heading(s):
    return re.sub(r"\s+", " ", OBSIDIAN_STRIP.sub(" ", s)).strip()


def frontmatter(path):
    """Parse the simple `key: value` lines of a leading YAML block.
    Block lists (`key:` then `  - item` lines, as Obsidian writes them) become `[a, b]`."""
    meta, key = {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        if f.readline().strip() != "---":
            return meta
        for line in f:
            if line.strip() == "---":
                break
            m = re.match(r"^(\w+):\s*(.*?)\s*$", line)
            if m:
                key = m.group(1)
                meta[key] = m.group(2).strip("'\"")
                continue
            m = re.match(r"^\s+-\s+(.*?)\s*$", line)
            if m and key:
                items = meta[key].strip("[]")
                meta[key] = "[" + ", ".join(filter(None, [items, m.group(1).strip("'\"")])) + "]"
    return meta


def render_text(s):
    """Approximate the plain text GitHub renders for a heading."""
    code = re.findall(r"`([^`]*)`", s)  # code spans keep their <...> text
    s = re.sub(r"`[^`]*`", "\0", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", s)
    for c in code:
        s = s.replace("\0", c, 1)
    s = s.replace("`", "")
    s = re.sub(r"(\*\*|__)(.+?)\1", r"\2", s)
    s = re.sub(r"(?<![\w*])([*_])(?!\s)(.+?)(?<!\s)\1(?![\w*])", r"\2", s)
    return s.strip()


def slugify(text):
    """github-slugger: lowercase, drop punctuation (keep letters/marks/numbers/_/-/space), space -> '-'."""
    out = []
    for ch in text.lower():
        cat = unicodedata.category(ch)
        if ch in "-_ " or cat[0] in "LMN":
            out.append(ch)
    return "".join(out).replace(" ", "-")


def headings(path):
    """[(level, text, slug, raw)] for real headings (outside frontmatter / fences / HTML comments).

    slug is GitHub's deduplicated anchor, raw the heading source as Obsidian sees it.
    """
    res, seen = [], {}
    fence, comment, front = None, False, False
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i == 0 and line.strip() == "---":
                front = True
                continue
            if front:
                front = line.strip() != "---"
                continue
            if comment:
                comment = "-->" not in line
                continue
            m = FENCE.match(line)
            if fence:
                if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not line.strip()[len(m.group(1)):].strip():
                    fence = None
                continue
            if m:
                fence = m.group(1)
                continue
            if line.lstrip().startswith("<!--") and "-->" not in line:
                comment = True
                continue
            h = HEAD.match(line)  # GitHub allows up to 3 leading spaces; notes are flush-left
            if not h:
                continue
            text = render_text(h.group(2))
            base = slugify(text)
            n = seen.get(base, 0)
            seen[base] = n + 1
            res.append((len(h.group(1)), text, base if n == 0 else f"{base}-{n}", h.group(2)))
    return res


def note_info(rel):
    path = os.path.join(PUB, rel)
    hs = headings(path)
    title, body = None, hs
    # Title: first heading written as # ___Title___
    if hs and hs[0][0] == 1 and hs[0][3].startswith("___"):
        title, body = hs[0][1], hs[1:]
    if not title:
        stem = os.path.splitext(os.path.basename(rel))[0]
        title = re.sub(r"^\d\d-\d\d?_", "", stem).replace("_", " ")
    title = re.sub(r"^\d{4}\s*-\s*\d\d\s*-\s*\d\d\s*", "", title).strip(" （）()") or title
    candidates = [h for h in body if h[1].lower() not in SKIP_SECTIONS and h[2]]
    for level in (1, 2):
        picked = [h for h in candidates if h[0] <= level]
        if len(picked) >= 3:
            break
    sections = picked[:MAX_SECTIONS]
    size = os.path.getsize(path)
    folder = rel.split("/")[0]
    m = re.match(r"(\d{4})", folder)
    md = re.match(r"(\d\d)-(\d\d?)_", os.path.basename(rel))
    date = f"{m.group(1)}-{md.group(1)}-{int(md.group(2)):02d}" if m and md else ""
    summary = frontmatter(path).get("summary", "")
    return dict(rel=rel, title=title, sections=sections, size=size, date=date, summary=summary)


def esc(s, limit=48):
    s = s if len(s) <= limit else s[: limit - 1].rstrip() + "…"
    return s.replace("[", r"\[").replace("]", r"\]").replace("<", r"\<")


def anchor(slug, raw):
    """GitHub slug if Obsidian resolves it to the same heading, else Obsidian's heading anchor."""
    obsidian = obsidian_heading(raw)
    if obsidian_heading(slug).lower() == obsidian.lower():
        return slug
    return obsidian.replace(" ", "%20")


def href(rel, anchor=None):
    # Keep CJK readable; only escape what breaks a Markdown link destination
    url = "../public/" + rel.replace(" ", "%20").replace("(", "%28").replace(")", "%29")
    return url + (f"#{anchor}" if anchor else "")


def note_line(n):
    meta = [n["date"]] if n["date"] else []
    if n["size"] < 200:
        meta.append("empty stub")
    line = f"- **[{esc(n['title'], 80)}]({href(n['rel'])})**"
    if meta:
        line += f" ({', '.join(meta)})"
    if n["summary"]:
        line += f" — {n['summary']}"
    if n["sections"]:
        line += "\n  - " + " · ".join(f"[{esc(t)}]({href(n['rel'], anchor(s, raw))})" for _, t, s, raw in n["sections"])
    return line


def main():
    all_notes = sorted(
        os.path.relpath(os.path.join(d, f), PUB)
        for d, _, fs in os.walk(PUB) for f in fs if f.endswith(".md")
    )
    all_notes = [r for r in all_notes if r != "README.md"]
    titles = {t[0]: t[1] for t in TOPICS}
    assigned = {}
    os.makedirs(OUT, exist_ok=True)
    readme_rows = []
    for key, name, intro, groups, related in TOPICS:
        lines = [f"# {name}", "", intro, "", "Back to [all indexes](README.md).", ""]
        count = 0
        for gname, pats in groups:
            rels = []
            for p in pats:
                hit = [r for r in all_notes if r.startswith(p)]
                assert hit, (key, p)
                rels += hit
            rels = sorted(dict.fromkeys(rels))
            if gname:
                lines += [f"## {gname}", ""]
            for r in rels:
                assert r not in assigned, (r, assigned.get(r), key)
                assigned[r] = key
                lines.append(note_line(note_info(r)))
                count += 1
            lines.append("")
        if related:
            lines += ["## See also", ""] + [f"- [{titles[r]}]({r}.md)" for r in related] + [""]
        with open(os.path.join(OUT, f"{key}.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")
        readme_rows.append(f"| [{name}]({key}.md) | {count} | {intro} |")
    missing = [r for r in all_notes if r not in assigned]
    assert not missing, missing
    readme = [
        "# Notebook Indexes", "",
        "Topic indexes (Maps of Content) for the notes in [`public/`](../public/).",
        "Each index links every note in the topic and its main sections.", "",
        "| Index | Notes | Covers |", "| --- | ---: | --- |", *readme_rows, "",
        "Generated by [`build_index.py`](build_index.py) from note headings and `summary:` frontmatter;",
        "rerun `python3 index/build_index.py` after adding notes.", "",
    ]
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))
    print(len(assigned), "notes in", len(TOPICS), "indexes")


if __name__ == "__main__":
    main()
