- [CANN atb llm 社区版资源下载](https://www.hiascend.com/developer/download/community/result?cann=7.0.0.beta1&product=5&model=25)
## Torch model dump data
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
## ait llm
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
## ait llm compare
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
## ait llm compare parallel tensors
  ```py
  import os
  import re
  import torch
  import numpy as np
  from llm.common.tool import TensorBinFile

  DEFAULT_MAPPING_DICT = {
      "embed_tokens": "0_GatherOperation",
      "input_layernorm": "0_RmsNormOperation",
      "self_attn.query_key_value": "1_LinearOperation",
      "self_attn.rotary_emb": "7_RopeOperation",
      "self_attn.o_proj": "12_ParallelLinearBase",
      "mlp.gate_up_proj.layernorm": "14_RmsNormOperation",
      # "mlp.down_proj": "",
      "mlp.act_fn": "15_MlpGateLayerBase/2_ActivationOperation",
      "mlp.gate_up_proj": "15_MlpGateLayerBase/4_ParallelLinearBase",
      "mlp": "15_MlpGateLayerBase",
  }


  def cosine_sim(left, right):
      left = left.ravel().astype("float32")
      right = right.ravel().astype("float32")
      return ((left / np.linalg.norm(left)) * (right / np.linalg.norm(right))).sum()


  def compare_tensor(gpu_path, npu_path):
      aa = np.load(gpu_path) if gpu_path.endswith("npy") else torch.load(gpu_path, map_location=torch.device("cpu")).float().cpu().numpy()
      bb = TensorBinFile(npu_path).get_data()
      return cosine_sim(aa, bb), aa, bb


  def compare_tensor_parallel_tensor(gpu_path, npu_pathes):
      gpu_data = np.load(gpu_path) if gpu_path.endswith("npy") else torch.load(gpu_path, map_location=torch.device("cpu")).float().cpu().numpy()
      npu_datas = [TensorBinFile(ii).get_data() for ii in npu_pathes]
      print(f">>>> gpu_data.shape: {gpu_data.shape}, npu_datas.shape: {[ii.shape for ii in npu_datas]}")

      gpu_data, npu_datas = gpu_data.squeeze(), [ii.squeeze() for ii in npu_datas]
      gpu_shape_prod, npu_shape_prod = np.prod(gpu_data.shape), sum([np.prod(ii.shape) for ii in npu_datas])
      if gpu_shape_prod != npu_shape_prod and gpu_shape_prod != np.prod(npu_datas[0].shape):
          message = f"Shape not matching: gpu: {gpu_data.shape}, npus: {[ii.shape for ii in npu_datas]}"
          print(f">>>> [ERROR] {message}")
          return None, gpu_data, npu_datas, message

      if gpu_data.shape == npu_datas[0].shape:
          npu_data = npu_datas[0]
      elif gpu_data.shape[0] != npu_datas[0].shape[0]:
          npu_data = np.concatenate(npu_datas, axis=0)
      elif gpu_data.shape[-1] != npu_datas[0].shape[-1]:
          npu_data = np.concatenate(npu_datas, axis=-1)
      else:
          message = f"Neither aixs 0 or -1: gpu: {gpu_data.shape}, npus: {[ii.shape for ii in npu_datas]}"
          print(f">>>> [ERROR] {message}")
          return None, gpu_data, npu_datas, message

      if gpu_data.shape != npu_data.shape:
          message = f"After reduce shape still not matching: gpu: {gpu_data.shape}, npus: {npu_data.shape}"
          return None, gpu_data, npu_data, message

      return cosine_sim(gpu_data, npu_data), gpu_data, npu_data, ""


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


  def gather_dump_data(data_path, split_index=-1):
      data_path_len = len(data_path)
      gathered_files = []
      for cur_path, dirs, file_names in os.walk(data_path):
          if len(file_names) == 0:
              continue
          if split_index < -1:
              cur_path = os.sep.join(cur_path.split(os.sep)[: split_index + 1])
          if len(cur_path) == data_path_len:
              continue
          gathered_files.append(cur_path[data_path_len:])
      return gathered_files


  def get_operation_path(operation, collected_operations):
      for ii in collected_operations:
          if ii.endswith(operation):
              return ii
      return ""


  def init_layer_dict_with_actual_path(npu_path, origin_dict):
      layer_path = None
      npu_path = os.path.abspath(npu_path)
      for ii in os.listdir(npu_path):
          if ii.endswith("_ZhiPuLayer"):
              layer_path = os.path.join(npu_path, ii)
              break
      if not layer_path:
          return origin_dict

      collected_npu_operations = gather_dump_data(layer_path, split_index=-2)  # get rid of `after`
      cur_dict = {kk: get_operation_path(vv, collected_npu_operations) for kk, vv in origin_dict.items() if vv}
      return {kk: vv[1:] for kk, vv in cur_dict.items() if kk and vv}


  def match_rule_modules(gpu_input_name, gpu_path, npu_path, mapping_dict):
      if not ".layers." in gpu_input_name:
          return None

      npu_op_name = None
      for kk in mapping_dict:
          if gpu_input_name.endswith(kk):
              npu_op_name = mapping_dict[kk]
              break
      if not npu_op_name:
          return None

      layer_id = int(gpu_input_name.split(".layers.")[-1].split(".")[0])
      npu_layer_name = str(layer_id + 1) + "_ZhiPuLayer"
      print(f">>>> gpu_input_name: {gpu_input_name}, layer_id: {layer_id}, npu_layer_name: {npu_layer_name}, npu_op_name: {npu_op_name}")

      cur_gpu_path = os.path.join(gpu_path, gpu_input_name)
      cur_npu_path = os.path.join(npu_path, npu_layer_name, npu_op_name, "after")
      print(f"     cur_gpu_path: {cur_gpu_path}")
      print(f"     cur_npu_path: {cur_npu_path}")
      if not os.path.exists(cur_npu_path):
          print(f"[ERROR] npu_layer_name {npu_layer_name} not exists in {npu_path}")
          return None

      gpu_inputs, gpu_outputs = [], []
      for ii in sorted(os.listdir(cur_gpu_path)):
          if "input" in ii:
              gpu_inputs.append(os.path.join(cur_gpu_path, ii))
          else:
              gpu_outputs.append(os.path.join(cur_gpu_path, ii))

      npu_path_len = len(npu_path) + 1  # get rid of ending `\`
      npu_inputs, npu_outputs = [], []
      for ii in sorted(os.listdir(cur_npu_path)):
          if "intensor" in ii:
              npu_inputs.append(os.path.join(cur_npu_path, ii)[npu_path_len:])
          else:
              npu_outputs.append(os.path.join(cur_npu_path, ii)[npu_path_len:])
      print(f"     gpu_inputs: {len(gpu_inputs)}, gpu_outputs: {len(gpu_outputs)}")
      print(f"     npu_inputs: {len(npu_inputs)}, npu_outputs: {len(npu_outputs)}")
      return list(zip(gpu_inputs, npu_inputs)) + list(zip(gpu_outputs, npu_outputs))


  def basic_compare_with_match_rule(gpu_path, npu_path, match_rule, mapping_dict=DEFAULT_MAPPING_DICT, token_id=0):
      origin_gpu_path_len, origin_npu_path_len, token_id = len(gpu_path), len(npu_path), str(token_id)
      gpu_path = gpu_path if os.path.basename(os.path.abspath(gpu_path)) == token_id else os.path.join(gpu_path, token_id)
      if all([re.match(r"\d_\d+$", ii) is not None for ii in os.listdir(npu_path)]):
          npu_pathes = [os.path.join(npu_path, ii, token_id) for ii in sorted(os.listdir(npu_path))]
      else:
          npu_path = npu_path if os.path.basename(os.path.abspath(npu_path)) == token_id else os.path.join(npu_path, token_id)
          npu_pathes = [npu_path]
      print(f">>>> gpu_path: {gpu_path}, npu_path: {npu_path}")

      mapping_dict = init_layer_dict_with_actual_path(npu_pathes[0], mapping_dict)
      print(f">>>> mapping_dict: {mapping_dict}")
      gathered_mappings = []
      for cur_gpu_file in os.listdir(gpu_path):
          mappings = match_rule(cur_gpu_file, gpu_path, npu_pathes[0], mapping_dict)
          print(f">>>> cur_gpu_file: {cur_gpu_file}, mappings: {mappings}")
          if mappings is None:
              continue
          gathered_mappings.extend(mappings)

      sort_key = lambda xx: [int(ii.split("_")[0]) for ii in xx[1].split(os.sep)[:2]]  # Sort by int index
      gathered_mappings = sorted(gathered_mappings, key=sort_key)
      print(f"\n>>>> gathered_mappings: {gathered_mappings}\n")

      gathered_results = []
      for cur_gpu_file, cur_npu_file in gathered_mappings:
          cur_gpu_path = os.path.join(gpu_path, cur_gpu_file)
          cur_npu_pathes = [os.path.join(ii, cur_npu_file) for ii in npu_pathes]
          print(f"cur_gpu_path: {cur_gpu_path}, cur_npu_pathes: {cur_npu_pathes}")

          if any([not os.path.exists(ii) for ii in cur_npu_pathes]):
              print("[Error] not exists:", cur_npu_pathes)
              continue
          if not os.path.exists(cur_gpu_path):
              print("[Error] not exists:", cur_gpu_path)
              continue
          cosine_sim, _, _, message = compare_tensor_parallel_tensor(cur_gpu_path, cur_npu_pathes)

          rr = {"NPU {}".format(id): ii[origin_npu_path_len:] for id, ii in enumerate(cur_npu_pathes)}
          rr.update({"GPU": cur_gpu_path[origin_gpu_path_len:], "Cosine": cosine_sim, "Error info": message})
          gathered_results.append(rr)
      return gathered_results


  if __name__ == "__main__":
      import argparse

      parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
      parser.add_argument("-g", "--gpu", type=str, default="/home/data/suwenyuan/dump_data/13965_cuda_0/0/", help="GPU dump data path")
      parser.add_argument("-n", "--npu", type=str, default="/home/data/shikang/test/code/dump/test/", help="NPU dump data path")
      parser.add_argument("-t", "--token_id", type=int, default=0, help="specify a single token id for comparing")
      parser.add_argument("-o", "--output", type=str, default="result.csv", help="csv output path")
      args = parser.parse_known_args()[0]

      # gg = "/home/leondgarse/workspace/zhipu_data/gpu_data/13965_cuda_0/0"
      # nn = "/home/leondgarse/workspace/zhipu_data/npu_data"
      csv_contents = basic_compare_with_match_rule(
          gpu_path=os.path.abspath(args.gpu),
          npu_path=os.path.abspath(args.npu),
          match_rule=match_rule_modules,
          mapping_dict=DEFAULT_MAPPING_DICT,
          token_id=args.token_id,
      )
      print("\n>>>> csv_contents:")
      print('\n'.join(['\n    '.join(['{'] + ['{}: {},'.format(kk, vv) for kk, vv in ii.items()]) + '\n},' for ii in csv_contents]))

      import pandas as pd

      output = (args.output + ".csv") if not args.output.endswith(".csv") else args.output
      pd.DataFrame(csv_contents).to_csv(output)
      print(f"\n>>>> Result saved to: {output}")
  ```
