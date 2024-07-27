# ___2021 - 08 - 13 Haiku___
***

## MNIST
  ```py
  from typing import Generator, Mapping, Tuple

  import haiku as hk
  import jax
  import jax.numpy as jnp
  import numpy as np
  import optax
  import tensorflow_datasets as tfds

  Batch = Mapping[str, np.ndarray]


  def net_fn(batch: Batch) -> jnp.ndarray:
      """Standard LeNet-300-100 MLP network."""
      x = batch["image"].astype(jnp.float32) / 255.
      mlp = hk.Sequential([
          hk.Flatten(),
          hk.Linear(300), jax.nn.relu,
          hk.Linear(100), jax.nn.relu,
          hk.Linear(10),
      ])
      return mlp(x)


  def load_dataset(
      split: str,
      *,
      is_training: bool,
      batch_size: int,
  ) -> Generator[Batch, None, None]:
      """Loads the dataset as a generator of batches."""
      ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
      if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
      ds = ds.batch(batch_size)
      return iter(tfds.as_numpy(ds))


  # Make the network and optimiser.
  net = hk.without_apply_rng(hk.transform(net_fn))
  print(hk.experimental.tabulate(net, columns=('module', 'owned_params', 'input', 'output', 'params_size'))({"image": jnp.ones([1, 32 * 32])}))
  opt = optax.adam(1e-3)

  # Training loss (cross-entropy).
  def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
      """Compute the loss of the network, including L2."""
      logits = net.apply(params, batch)
      labels = jax.nn.one_hot(batch["label"], 10)

      l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
      softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
      softmax_xent /= labels.shape[0]

      return softmax_xent + 1e-4 * l2_loss

  # Evaluation metric (classification accuracy).
  @jax.jit
  def accuracy(params: hk.Params, batch: Batch) -> jnp.ndarray:
      predictions = net.apply(params, batch)
      return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])

  @jax.jit
  def update(
      params: hk.Params,
      opt_state: optax.OptState,
      batch: Batch,
  ) -> Tuple[hk.Params, optax.OptState]:
      """Learning rule (stochastic gradient descent)."""
      grads = jax.grad(loss)(params, batch)
      updates, opt_state = opt.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      return new_params, opt_state

  # We maintain avg_params, the exponential moving average of the "live" params.
  # avg_params is used only for evaluation (cf. https://doi.org/10.1137/0330046)
  @jax.jit
  def ema_update(params, avg_params):
      return optax.incremental_update(params, avg_params, step_size=0.001)

  # Make datasets.
  train = load_dataset("train", is_training=True, batch_size=1000)
  train_eval = load_dataset("train", is_training=False, batch_size=10000)
  test_eval = load_dataset("test", is_training=False, batch_size=10000)

  # Initialize network and optimiser; note we draw an input to get shapes.
  params = avg_params = net.init(jax.random.PRNGKey(42), next(train))
  opt_state = opt.init(params)

  # Train/eval loop.
  for step in range(10001):
      if step % 1000 == 0:
          # Periodically evaluate classification accuracy on train & test sets.
          train_accuracy = accuracy(avg_params, next(train_eval))
          test_accuracy = accuracy(avg_params, next(test_eval))
          train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
          print(f"[Step {step}] Train / Test accuracy: " f"{train_accuracy:.3f} / {test_accuracy:.3f}.")

      # Do SGD on a batch of training examples.
      params, opt_state = update(params, opt_state, next(train))
      avg_params = ema_update(params, avg_params)
  ```
## Cifar10
  ```py
  import haiku as hk
  import jax
  import jax.numpy as jnp
  import optax
  import jmp
  import tensorflow_datasets as tfds
  from tqdm import tqdm

  mp_policy = jmp.get_policy('p=f32,c=f32,o=f32')
  hk.mixed_precision.set_policy(hk.nets.ResNet50, mp_policy)

  def load_cifar10(batch_size=1024, image_shape=(32, 32)):
      import tensorflow_datasets as tfds
      AUTOTUNE = tf.data.experimental.AUTOTUNE

      if image_shape[:2] == (32, 32):
          preprocessing = lambda data: (tf.cast(data["image"], tf.float32) / 255.0, data["label"])
      else:
          preprocessing = lambda data: (tf.image.resize(data["image"], image_shape[:2]) / 255.0, data["label"])
          # preprocessing = lambda data: (tf.transpose(tf.image.resize(data["image"], image_shape[1:]) / 255.0, [2, 0, 1]), data["label"])
      dataset = tfds.load("cifar10", split="train").map(preprocessing, num_parallel_calls=AUTOTUNE)
      dataset = dataset.cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)
      return dataset


  def loss(params, state, x_input, y_true) -> jnp.ndarray:
      """Compute the loss of the network, including L2."""
      logits, state = model.apply(params, state, None, x_input) # The third one is RNG
      labels = jax.nn.one_hot(y_true, logits.shape[-1])

      softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
      softmax_xent /= labels.shape[0]

      return softmax_xent, state

  num_classes = 10
  model = hk.transform_with_state(lambda xx, is_training=True: hk.nets.ResNet50(num_classes)(xx, is_training=is_training))

  dummy_x = jnp.ones([1, 224, 224, 3])
  params, state = model.init(jax.random.PRNGKey(42), xx=dummy_x)
  opt = optax.adam(1e-3)
  opt_state = opt.init(params)

  train_ds = load_cifar10(batch_size=32, image_shape=(224, 224))
  steps_per_epoch = len(train_ds)
  for epoch in range(2):
      train = train_ds.as_numpy_iterator()
      for (xx, yy) in tqdm(train, total=steps_per_epoch):
          grads, new_state = jax.grad(loss, has_aux=True)(params, state, xx, yy)
          updates, new_opt_state = opt.update(grads, opt_state)
          new_params = optax.apply_updates(params, updates)
          params, opt_state, state = new_params, new_opt_state, new_state
  ```
## Tests
  - Model summary
  ```py
  import haiku as hk
  import jax.numpy as jnp
  ff = hk.transform(lambda xx: hk.nets.MLP([300, 100, 10])(xx))
  print(hk.experimental.tabulate(ff, columns=('module', 'owned_params', 'input', 'output'))(jnp.ones([1, 32 * 32])))

  net = hk.transform(lambda xx: hk.nets.ResNet50(1000)(xx, is_training=True))
  print(hk.experimental.tabulate(net, columns=('module', 'owned_params', 'input', 'output'))(jnp.ones([1, 224, 224, 3])))
  ```
  - graphviz
  ```py
  dot = hk.experimental.to_dot(f)(x)
  dot = hk.experimental.to_dot(f.apply)(params, None, x)

  import graphviz
  graphviz.Source(dot)
  ```
***
