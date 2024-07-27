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
