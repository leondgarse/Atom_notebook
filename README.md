# Personal notebook
***
# Don't Read me!
## Add a note
1. Create `public/<year>/MM-DD_<topic>.md` with frontmatter and a title:
   ```md
   ---
   tags: [life-languages, travel, shandong]
   summary: One line on what the note covers.
   ---
   # ___2021 - 11 - 14 Title___
   ***
   ```
   The first tag is the topic key with `-` (`life-languages` for `life_languages`).
2. Add the note's path prefix (e.g. `"2021/11-14_shandong"`) to its topic in `TOPICS` in `index/build_index.py`.
3. Optional: hand-pick its Related links in `RELATED` in `index/build_related.py`; otherwise they are computed.
4. Run `index/update.sh`.

## Remove a note
1. Delete the file.
2. Remove its prefix from `TOPICS` in `index/build_index.py`, and from `RELATED` in `index/build_related.py` (as a key or a link).
3. Run `index/update.sh`.

## Generated files
Don't hand-edit them; `index/update.sh` rewrites all of them.
```shell
python3 index/build_related.py --write   # "# Related" section at the end of each note
python3 index/build_index.py             # index/*.md topic indexes
cd public && python3 2017/folder_toc.py  # writes public/README.md
```
