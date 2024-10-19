## Torch FX
```py
from transformers import AutoConfig, AutoModelForCausalLM
cc = AutoConfig.from_pretrained()
mm = AutoModelForCausalLM.from_config(cc)

import torch
import functorch

tt = functorch.make_fx(mm)(torch.ones([1, 32]).long())
print(tt.graph)
print(tt.code)
```
## LLM parse
```py
import transformers

cc = transformers.models.llama.LlamaConfig()
cc.num_hidden_layers = 4
mm = transformers.AutoModelForCausalLM.from_config(cc)

from msit_llm.transform.model_parser import parser
rr = parser.build_model_tree(mm)

import json
json.dump(rr, open(rr['name'] + ".json", "w"))
```
***

IELTS writing task 1:
The sheet below shows the percentage of students at one adult education center taking various courses offered in 1985 and this year. Summarise the information by selecting and reporting the main features, and make comparisons where relevant.

| Year      | Fitness and dance | Art | Modern language(French/Spanish) | Local history | Music appreciation | IT  | Cookery |
| --------- | ----------------- | --- | ------------------------------- | ------------- | ------------------ | --- | ------- |
| 1985      | 22%               | 17% | 24%                             | 15%           | 8%                 | 0%  | 14%     |
| This Year | 22%               | 15% | 17%                             | 0%            | 0%                 | 27% | 19%     |

Here's my essay:
The sheet illustrates the courses students taken in one adult education, and compares the percentage of students taking various courses in 1985 and current year.

In 1985, learning a modern language including French and Spanish was most popular, at 24%. Meanwhile, fitness and dance course was also at a same level of popularity, both reached above 20%. Following that, art, local history, cookery were less taken, all reaching around 15%. Then the third level preferred course was music appreciation, which is taken by only 8% students.

In contrast, the most significant change is this year is that, IT course reached its peak at 27%, which was even not exist in 1985. At the same time, courses like fitness and dance, cookery, art just kept a same level as previously. However, was the most popular course in 1985 at 24%, it dropped to 17% this year, losing its top position to IT. More significantly, other courses like local history and music appreciation suffered a complete elimination.

Overall, the most dramatic change in courses popularity is the raising of IT. Practical skills like cookery also keeps a noticable increasing, whereas traditional classes including local history and music appreciation suffered a dramatic drop. Other courses like art and modern language just fluctuated a bit. Especially modern languages was the most popular 1985, but lost its top position to IT.

git revert 911cac50d583d6f3120cd79cc57f73ea029006d6
git revert 9059492833a5663b7e852a50fb55ea6cdfa78f2c
git diff --name-only --diff-filter=U | xargs -I {} git rm {} && git revert --continue

git revert f14c06d4d8288698f835ccda251b05454caf1ed8
git diff --name-only --diff-filter=U | xargs -I {} git rm {} && git revert --continue

git revert eadb60fb0abcb08fc20af0ee240764b28f36e845
git revert 5ecdfc627f63cdc8d0d0675e7bd3c2ae166bee07
git diff --name-only --diff-filter=U | xargs -I {} git rm {} && git revert --continue

git revert 68ab20ad2e4a425b7484586ba4d68dcfae4395d8
git diff --name-only --diff-filter=U | xargs -I {} git rm {} && git revert --continue

git revert 73c0d8b2484c32a738b1849d3468157251e78518
git diff --name-only --diff-filter=U | xargs -I {} git rm {} && git revert --continue

git revert 04ec7807f1c082c306f1dc4a005385969159587f
git diff --name-only --diff-filter=U | xargs -I {} git rm {} && git revert --continue

git revert d0f0f5170bb7b144e28b8a95f89af9e4140ca061
git revert 744ff8d070a9b8788ba0aae4efaddafb6cfde886
git diff --name-only --diff-filter=U | xargs -I {} git rm {} && git revert --continue

git revert 7e1cb3157c94c9ecedceaa8c6df331b4713b8deb
