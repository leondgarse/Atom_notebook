# h5py
  ```py
  import h5py
  from keras_cv_attention_models import backend, mlp_mixer

  HDF5_OBJECT_HEADER_LIMIT = 64512

  def save_subset_weights_to_hdf5_group(group, weight_names, weight_values):
      """Save top-level weights of a model to a HDF5 group."""
      save_attributes_to_hdf5_group(group, "weight_names", weight_names)
      for name, val in zip(weight_names, weight_values):
          param_dset = group.create_dataset(name, val.shape, dtype=val.dtype)
          if not val.shape:
              # scalar
              param_dset[()] = val
          else:
              param_dset[:] = val

  def save_attributes_to_hdf5_group(group, name, data):
      """Saves attributes (data) of the specified name into the HDF5 group.

      This method deals with an inherent problem of HDF5 file which is not
      able to store data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

      Args:
          group: A pointer to a HDF5 group.
          name: A name of the attributes to save.
          data: Attributes data to store.

      Raises:
        RuntimeError: If any single attribute is too large to be saved.
      """
      # Check that no item in `data` is larger than `HDF5_OBJECT_HEADER_LIMIT`
      # because in that case even chunking the array would not make the saving
      # possible.
      bad_attributes = [x for x in data if len(x) > HDF5_OBJECT_HEADER_LIMIT]

      # Expecting this to never be true.
      if bad_attributes:
          raise RuntimeError(
              "The following attributes cannot be saved to HDF5 file because "
              f"they are larger than {HDF5_OBJECT_HEADER_LIMIT} "
              f"bytes: {bad_attributes}"
          )

      data_npy = np.asarray(data)

      num_chunks = 1
      chunked_data = np.array_split(data_npy, num_chunks)

      # This will never loop forever thanks to the test above.
      while any([x.nbytes > HDF5_OBJECT_HEADER_LIMIT for x in chunked_data]):
          num_chunks += 1
          chunked_data = np.array_split(data_npy, num_chunks)

      if num_chunks > 1:
          for chunk_id, chunk_data in enumerate(chunked_data):
              group.attrs["%s%d" % (name, chunk_id)] = chunk_data
      else:
          group.attrs[name] = data


  def save_weights_to_hdf5_group(h5_file, model):
      """Saves the weights of a list of layers to a HDF5 group."""

      save_attributes_to_hdf5_group(h5_file, "layer_names", [layer.name.encode("utf8") for layer in model.layers])
      h5_file.attrs["backend"] = backend.backend().encode("utf8")
      for layer in sorted(model.layers, key=lambda x: x.name):
          layer_group = h5_file.create_group(layer.name)
          weight_names = [ww.name.encode("utf8") for ww in layer.weights]
          weight_values = layer.get_weights()
          save_subset_weights_to_hdf5_group(layer_group, weight_names, weight_values)

  model = mlp_mixer.MLPMixerB16(input_shape=(3, 224, 224))
  with h5py.File("foo.h5", "w") as h5_file:
      save_weights_to_hdf5_group(h5_file, model)
  ```
***

# Keras delete layer
```py
mm = keras.applications.ResNet50(weights=None)
aa = mm.get_layer('conv5_block3_2_conv')
bb = mm.get_layer('conv5_block3_2_bn')
for out_node in bb.outbound_nodes:
    for id, in_node in enumerate(out_node.layer.inbound_nodes):
        if in_node.layer.input.node.layer is bb:
            print(in_node.layer.name)
            out_node.layer.inbound_nodes[id] = bb.inbound_nodes[0]

for id, out_node in enumerate(aa.outbound_nodes):
    if out_node.layer is bb:
        print(out_node.layer.name)
        aa.outbound_nodes[id] = bb.outbound_nodes[0]

for id, ii in enumerate(mm.layers):
    if ii.name == bb.name:
        print(ii.name, id)
        mm._self_tracked_trackables.pop(id)

mm.summary()
dd = keras.models.model_from_json(mm.to_json())
dd.summary()
```
```py
from keras.engine.node import Node

mm = keras.applications.ResNet50(weights=None)
aa = mm.get_layer('conv5_block3_2_conv')
cc = keras.layers.Dense(512)
ee = Node(cc)

cc.outbound_nodes.append(aa.outbound_nodes[0])
aa.outbound_nodes[0] = cc.inbound_nodes[0]

cc.inbound_nodes.append(bb.inbound_nodes[0])
cc.outbound_nodes.append(bb.outbound_nodes[0])
```
# Downloader
```py
import os
import requests
from tqdm import tqdm
from urllib.request import urlopen, Request

def download_url_to_file(url, dst=None):
    CHUNK_SIZE = 8192
    dst = os.path.splitext(url)[-1] if dst is None else dst
    dst = os.path.expanduser(dst)
    dst_name = os.path.basename(dst)
    response = requests.get(url, headers={"User-Agent": "torch.hub"})

    if not response.ok:
        print(dst_name + " Not found")
        return 0

    print(dst_name + " found")
    req = Request(url, headers={"User-Agent": "torch.hub"})
    meta = urlopen(req).info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    else:
        # file_size = None
        print("file_size Not known")
        return 0

    first_byte = os.path.getsize(dst) if os.path.exists(dst) else 0
    if first_byte >= file_size:
        return file_size

    first_byte = (first_byte // CHUNK_SIZE) * CHUNK_SIZE
    headers = {"User-Agent": "torch.hub", "range": "bytes=%s-%s" % (first_byte, file_size)}
    req = requests.get(url, headers=headers, stream=True)
    try:
        ff = open(dst, "ab")
        if ff.tell() != first_byte:
            ff.seek(first_byte)
        with tqdm(total=file_size, initial=first_byte, unit='B', unit_scale=True, unit_divisor=1024, ascii=" >=", desc=dst_name) as pbar:
            for chunk in req.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    ff.write(chunk)
                    pbar.update(len(chunk))
                else:
                    break
    finally:
        ff.close()
    return file_size
```
```py
import os
import requests
from tqdm import tqdm
from urllib.request import urlopen, Request


```
```py
import shutil
import tempfile
from urllib.request import urlopen, Request
from tqdm import tqdm

url = 'https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_lite_small_imagenet.h5'

try:
    headers = {"User-Agent": "torch.hub", "range": "bytes=%s-%s" % (25886720, file_size)}
    req = requests.get(url, headers=headers, stream=True)
    with open(dst, "ab") as ff, tqdm(total=file_size, disable=False, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
        for chunk in req.iter_content(chunk_size=8192):
            # buffer = u.read(8192)
            if len(chunk) == 0:
                break
            ff.write(chunk)
            pbar.update(len(chunk))

    f.close()
    shutil.move(f.name, dst)
finally:
    f.close()
    if os.path.exists(f.name):
        os.remove(f.name)
```
