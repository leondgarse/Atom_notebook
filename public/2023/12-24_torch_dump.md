```py
import numpy as np

def dump_output_hook(name):
    cur_token_id = 0

    def hook_func(module, inputs, outputs):
        # inputs is a tuple, and outputs tensor
        nonlocal cur_token_id

        ait_dump_path = os.getenv("AIT_DIALOG_DUMP_PATH", os.getenv("AIT_DUMP_PATH", "")) or ""
        cur_pid = str(os.getpid())
        output_path = os.path.join(ait_dump_path, cur_pid, str(cur_token_id))
        if not os.path.exists(output_path):
            os.makedirs(output_path, mode=0o750, exist_ok=True)
        output_file = os.path.join(output_path, name + "output.npy")
        print(name, [ii.shape for ii in inputs], outputs.shape)
        np.save(output_file, outputs.detach().cpu().numpy())
        cur_token_id += 1
    return hook_func

def register_forward_hook(model):
    used_names = {}
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        used_names[class_name] = used_names.get(class_name, -1) + 1  # starts from 0
        name += "-{}-{}-".format(class_name, used_names[class_name])  # seldom should a - exist in module name

        cur_forward_hook = dump_output_hook(name=name)
        module.register_forward_hook(cur_forward_hook)

import os
os.environ['KECAM_BACKEND'] = 'torch'

import torch, kecam
model = kecam.models.LLaMA2_15M()
register_forward_hook(model)
model(torch.ones([1, 12])).shape
```
