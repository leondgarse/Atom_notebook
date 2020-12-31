#!/usr/bin/python3
import os
import sys
import argparse


def is_file_endswith_suffixs(file_name, suffix_list=[]):
    for suffix in suffix_list:
        if file_name.endswith(suffix):
            return True
    return False


def folder_toc(path_name, retract_level, output_file=None, excluded_suffix=[]):
    # print("path_name = %s" % path_name)
    for sub_item in sorted(os.listdir(path_name)):
        if is_file_endswith_suffixs(sub_item, excluded_suffix):
            continue

        # print("sub_item = %s" % sub_item_path)
        sub_item_path = os.path.join(path_name, sub_item)
        outputline = "  " * retract_level + "- [%s](%s)" % (sub_item, sub_item_path)
        if output_file != None:
            output_file.write(outputline + "\n")
        else:
            print(outputline)

        if os.path.isdir(sub_item_path):
            folder_toc(sub_item_path, retract_level + 1, output_file, excluded_suffix)


def folder_toc_2_file(path_name, retract_level, output_file_name, excluded_suffix=[]):
    with open(output_file_name, "w") as ff:
        folder_toc(path_name, retract_level, ff, excluded_suffix)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_path", type=str, help="The path used to generate markdown TOC", default="./")
    parser.add_argument("-o", "--output", type=str, help="Output markdown file name", default="./readme.md")
    parser.add_argument(
        "-e",
        "--excluded_suffix",
        type=str,
        nargs="*",
        default=[".jpg", ".png", ".gif", ".svg", ".jpeg"],
        help="File suffixes that are exluded in TOC",
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    print("input_path = %s, output = %s, excluded_suffix = %s" % (args.input_path, args.output, args.excluded_suffix))
    folder_toc_2_file(args.input_path, 0, args.output, args.excluded_suffix)
