#!/bin/sh
# Regenerate Related sections, topic indexes and public/README.md after adding or removing notes
set -e
cd "$(dirname "$0")/.."
python3 index/build_related.py --write
python3 index/build_index.py
(cd public && python3 2017/folder_toc.py)
