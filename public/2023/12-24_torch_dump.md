- [CANN atb llm 社区版资源下载](https://www.hiascend.com/developer/download/community/result?cann=7.0.0.beta1&product=5&model=25)
- Torch model dump data
  ```py
  import numpy as np

  def dump_output_hook(name):
      cur_token_id = 0

      def hook_func(module, inputs, outputs):
          # inputs is a tuple, and outputs tensor
          nonlocal cur_token_id

          ait_dump_path = os.getenv("AIT_DIALOG_DUMP_PATH", os.getenv("AIT_DUMP_PATH", "")) or ""unset
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
- ait llm
  ```sh
  python -c 'import torch; print(torch.compiled_with_cxx11_abi())'

  wget https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231213/ait-0.0.1-py3-none-linux_aarch64.whl
  wget https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231226/ABI0/ait_llm-0.1.0-py3-none-linux_aarch64.whl
  pip install ait-0.0.1-py3-none-linux_aarch64.whl
  pip install ait_llm-0.1.0-py3-none-linux_aarch64.whl

  wget https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231213/ait-0.0.1-py3-none-linux_x86_64.whl
  wget https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240202/ait_llm-0.2.1-py3-none-linux_x86_64.whl
  pip install ait-0.0.1-py3-none-linux_x86_64.whl
  pip install ait_llm-0.2.1-py3-none-linux_x86_64.whl
  ```
- ait llm compare
  ```py
  import os
  import numpy as np
  from llm.common.tool import TensorBinFile

  gg = "/root/GPU-dump/dumpRes/1806192_cuda0/0"
  nn = "/root/foo/atb_temp/tensors/2098_2098/0"

  dd = {
      "input_layernorm": "0_RmsNormOperation",
      "mlp": "",
      "mlp.dense_4h_to_h": "",
      "mlp.dense_h_to_4h": "",
      "post_attention_layernorm": "7_RmsNormOperation",
      "self_attention": "",
      "self_attention.core_attention": "4_SelfAttentionOperation",
      "self_attention.dense": "5_ParallelLinearBase",
      "self_attention.query_key_value": "1_LinearOperation",
  }

  # 2_PositionEmbeddingTgi
  # 3_ReshapeAndCacheOperation
  # 4_SelfAttentionOperation
  #
  # 6_ElewiseOperation
  # 8_MlpGateLayerBase
  # 9_ElewiseOperation


  def get_all_shape(path):
      rr = []
      for cur, dirs, files in os.walk(path):
          for file in files:
              file_path = os.path.join(cur, file)
              rr.append({"path": file_path.replace(path, ""), "shape": TensorBinFile(file_path).get_data().shape})
      return rr


  def cosine_sim(left, right):
      left = left.ravel().astype('float32')
      right = right.ravel().astype('float32')
      return ((left / np.linalg.norm(left)) * (right / np.linalg.norm(right))).sum()

  def compare_tensor(gpu_path, npu_path):
      aa = np.load(gpu_path) if gpu_path.endswith("npy") else torch.load(gpu_path, map_location=torch.device('cpu')).float().cpu().numpy()
      bb = TensorBinFile(npu_path).get_data()
      return cosine_sim(aa, bb), aa, bb


  def match_rule_layers(gpu_input_name, gpu_path=None, npu_path=None):
      if not gpu_input_name.startswith("root.transformer.encoder.layers"):
          return None
      bb = gpu_input_name.replace("root.transformer.encoder.layers.", "")
      if "." in bb:
          return None
      tt = str(int(bb) + 1) + "_DecoderPALayer"

      npu_pth = os.path.join(tt, "after", "outtensor0.bin")
      gpu_pth = os.path.join(gpu_input_name, "output_exec1_0")
      return [[gpu_pth, npu_pth]]


  def match_rule_modules(gpu_input_name, gpu_path, npu_path):
      if not gpu_input_name.startswith("root.transformer.encoder.layers"):
          return None
      bb = gpu_input_name.replace("root.transformer.encoder.layers.", "")
      if "." not in bb:
          return None

      split_name = bb.split(".")
      layer_name = str(int(split_name[0]) + 1) + "_DecoderPALayer"
      gpu_op_name = split_name[-1]

      if gpu_op_name not in dd:
          return None
      npu_op_name = dd[gpu_op_name]
      op_path = os.path.join(npu_path, layer_name, npu_op_name, "after")
      if not os.path.exists(op_path):
          return None

      gpu_path = os.path.join(gpu_path, gpu_input_name)
      gpu_inputs, gpu_outputs = [], []
      for ii in sorted(os.listdir(gpu_path)):
          if "input" in ii:
              gpu_inputs.append(os.path.join(gpu_path, ii))
          else:
              gpu_outputs.append(os.path.join(gpu_path, ii))

      npu_inputs, npu_outputs = [], []
      for ii in sorted(os.listdir(op_path)):
          if "intensor" in ii:
              npu_inputs.append(os.path.join(op_path, ii))
          else:
              npu_outputs.append(os.path.join(op_path, ii))
      return list(zip(gpu_inputs, npu_inputs)) + list(zip(gpu_outputs, npu_outputs))


  def basic_compare_with_match_rule(gpu_path, npu_path, match_rule):
      rr = []
      for cur_gpu_file in os.listdir(gpu_path):
          mappings = match_rule(cur_gpu_file, gpu_path, npu_path)
          if mappings is None:
              continue

          for cur_gpu_file, cur_npu_file in mappings:
              cur_npu_path = os.path.join(npu_path, cur_npu_file)
              cur_gpu_path = os.path.join(gpu_path, cur_gpu_file)

              if not os.path.exists(cur_npu_path):
                  print("Error:", cur_npu_path)
                  continue
              if not os.path.exists(cur_gpu_path):
                  print("Error:", cur_gpu_path)
                  continue
              cosine_sim = compare_tensor(cur_gpu_path, cur_npu_path)[0]
              rr.append({"NPU": cur_npu_path, "GPU": cur_gpu_path, "Cosine": cosine_sim})
      return rr

  basic_compare_with_match_rule('/root/GPU-dump/dump/1856019_cuda0/0/', '/root/foo/atb_temp/tensors/2098_2098/0/', match_rule_layers)
  basic_compare_with_match_rule('/root/GPU-dump/dump/1856019_cuda0/0/', '/root/foo/atb_temp/tensors/2098_2098/0/', match_rule_modules)
  ```
