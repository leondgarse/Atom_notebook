#!/usr/bin/env python3
"""Write a Markdown file tree of a folder, e.g. `cd public && python3 2017/folder_toc.py`."""
import os
import sys
import argparse


def is_file_endswith_suffixs(file_name, suffix_list=[]):
    return file_name.lower().endswith(tuple(suffix.lower() for suffix in suffix_list))


def link_path(path):
    # Keep CJK readable; only escape what breaks a Markdown link destination
    return path.replace(" ", "%20").replace("(", "%28").replace(")", "%29")


def folder_toc(path_name, retract_level, link_base, excluded_suffix=[], excluded_files=[]):
    lines = []
    for sub_item in sorted(os.listdir(path_name)):
        sub_item_path = os.path.join(path_name, sub_item)
        if sub_item.startswith(".") or is_file_endswith_suffixs(sub_item, excluded_suffix):
            continue
        if os.path.abspath(sub_item_path) in excluded_files:
            continue

        sub_lines = []
        if os.path.isdir(sub_item_path):
            sub_lines = folder_toc(sub_item_path, retract_level + 1, link_base, excluded_suffix, excluded_files)
            if not sub_lines:  # e.g. images/ folders, whose files are all excluded
                continue

        link = os.path.relpath(sub_item_path, link_base).replace(os.sep, "/")
        link = link if link.startswith("../") else "./" + link
        lines.append("  " * retract_level + "- [%s](%s)" % (sub_item, link_path(link)))
        lines.extend(sub_lines)
    return lines


def folder_toc_2_file(path_name, retract_level, output_file_name, excluded_suffix=[]):
    # Links are relative to the output file, so -i and -o can point anywhere
    link_base = os.path.dirname(os.path.abspath(output_file_name))
    lines = folder_toc(path_name, retract_level, link_base, excluded_suffix, [os.path.abspath(output_file_name)])
    with open(output_file_name, "w", encoding="utf-8") as ff:
        ff.write("\n".join(lines) + "\n")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_path", type=str, help="The path used to generate markdown TOC", default="./")
    parser.add_argument("-o", "--output", type=str, help="Output markdown file name", default="./README.md")
    parser.add_argument(
        "-e",
        "--excluded_suffix",
        type=str,
        nargs="*",
        default=[".jpg", ".png", ".gif", ".svg", ".jpeg"],
        help="File suffixes that are exluded in TOC, case-insensitive. Hidden files are always excluded",
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    print("input_path = %s, output = %s, excluded_suffix = %s" % (args.input_path, args.output, args.excluded_suffix))
    folder_toc_2_file(args.input_path, 0, args.output, args.excluded_suffix)
