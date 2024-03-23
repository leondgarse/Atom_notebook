```py
import os
from unicodedata import normalize

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import ujson
from rich import progress

def split_txt_cropus_to_chunk_data(
    texts: list, batch_size: int = 512**2, max_len: int = 512, window_size: int = 2
) -> list:

    buffer, buffer_len = [], 0
    chunk_data = []

    for i, line in enumerate(texts):
        buffer_len += len(line)
        buffer.append(line)

        if buffer_len >= batch_size or i == len(texts) - 1:
            buffer_txt = "".join(buffer)

            # - window_size为滑动窗口，这样每个窗口都包含有window_size个上文
            for i in range(0, len(buffer_txt), max_len - window_size):

                chunk_data.append("".join(buffer_txt[i : i + max_len]))

            buffer, buffer_len = [], 0

    return chunk_data


import os, json
data_path = os.path.expanduser("~/workspace/datasets/wikipedia-cn-20230720-filtered.json")
with open(data_path, "r", encoding="utf-8") as f:
    texts = [ii["completion"] + "<|im_end|>" for ii in json.load(f)]

max_len, window_size = 512, 2
target_buffer_len, cur_buffer, gathered = max_len ** 2, "", []
for id, text in enumerate(texts):
    cur_buffer += text
    if len(cur_buffer) >= target_buffer_len or id == len(texts) - 1:
        gathered.extend([cur_buffer[ss: ss + max_len] for ss in range(0, len(cur_buffer), max_len - window_size)])
        cur_buffer = 0

chunk_data = split_txt_cropus_to_chunk_data(lines)
tb = pa.Table.from_arrays([pa.array(chunk_data)], names=["text"])
pq.write_table(
    table=tb,
    where=output_file,
    row_group_size=50000,
    data_page_size=50000,
)
```
