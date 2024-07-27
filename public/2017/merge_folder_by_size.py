#! /usr/bin/env python

import os
import sys
import argparse


def merge_folder_by_size_2(aa, bb, threash=0):
    moved = []
    not_moved = []
    for ff in os.listdir(bb):
        ss = os.path.join(bb, ff)
        dd = os.path.join(aa, ff)
        if os.path.exists(dd) and os.path.getsize(ss) - os.path.getsize(dd) < threash:
            not_moved.append(ff)
            print(
                "File not move: {}, size in {}: {:.2f}, size in {}: {:.2f}".format(
                    ff, aa, os.path.getsize(dd), bb, os.path.getsize(ss)
                )
            )
            continue

        moved.append(ff)
        os.rename(ss, dd)

    print("Source: {}, dist: {}, Moved file size: {}, Not moved file size: {}\n".format(bb, aa, len(moved), len(not_moved)))
    return moved, not_moved


def merge_folder_by_size_list(folder_list, threash):
    print("Merge {} to {}\n".format(folder_list[1:], folder_list[0]))
    aa = folder_list[0]
    moved = {}
    not_moved = {}
    for bb in folder_list[1:]:
        mm, nn = merge_folder_by_size_2(aa, bb, threash)
        moved[bb] = mm
        not_moved[bb] = nn

    print("All not moved:\n{}\n".format(not_moved))
    return moved, not_moved


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_list", type=str, nargs="+", help="Folder list will be merged")
    parser.add_argument(
        "--threash", type=int, default=0, help="File in dist is not <threash> bigger than source will not be merged"
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    merge_folder_by_size_list(args.folder_list, args.threash)

'''
mkdir -p test1 test2 test3
cd test1
touch aa bb cc dd ee ff
echo 'aa' > aa
echo 'bb' > bb

cd ../test2
touch aa bb cc dd ee ff
echo 'cc' > cc
echo 'dd' > dd

cd ../test3
touch aa bb cc dd ee ff
echo 'ee' > ee
echo 'ff' > ff

cd ../
tree test1 test2 test3

merge_folder_by_size.py test1 test2 test3

cd test2
echo 'aabbccdd' > aa
echo 'aabbccddeeffgghhiijj' > bb
cd ../

merge_folder_by_size.py test1 test2 test3 --threash 10
merge_folder_by_size.py test1 test2 test3 --threash 5
merge_folder_by_size.py test1 test2 test3 --threash -5
merge_folder_by_size.py test1 test2 test3 --threash -10

rm test1 test2 test3 -r
'''
