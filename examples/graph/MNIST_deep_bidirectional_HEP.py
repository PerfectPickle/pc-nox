"""MNIST_deep_bidirectional_HEP.py

This example demonstrates the effectiveness of Highway Error Propagation + rms activity normalisation in solving PCN depth scaling, and it does so bidirectionally.
The model has 65 hidden layers. This depth is not chosen for performance, rather to demonstrate successful bidirectional inference/learning at depth.
The model has checkpoints for both top-down and bottom-up pathways every 5 nodes, so in this both pathways share checkpoint nodes (apart from the highest hidden node which is checkpoint only for top-down).
After 10 epochs, the model achieves 93.7% classification test accuracy and clearly correct (easily identifiable) label to image generation for all digits.

Notes:
A bottom up precision of 1e-4 was found to yield better generative performance than 1e-5, despite needing a couple more epochs to reach good classification performance. A bu_precision of 1e-3 however, resulted in very poor performance.
The learning rates used, namely activity_lr of 5e-3, and param_lr of 6e-5, worked well for tanh, but not for gelu.
"""
from array import array
from os.path import join
import struct
import jax
import jax.numpy as jnp
import numpy as np
from pc_nox.models.graph import GraphModel, GraphModelConfig, NodeSpec
from pc_nox.models.graph.runners import make_train_step, make_eval_step, zero_states
from pc_nox.models.graph.optim_utils import grouped_optimiser

import optax
import equinox as eqx


import matplotlib.pyplot as plt
import numpy as np


def save_eval_samples(
    model,
    eval_step,
    X_test,
    Y_test,
    num_samples=15,
    filename="eval_samples.png",
):
	"""Evaluates a batch of test samples, displays images with predicted vs true labels,

	and saves the result to a PNG or JPEG file.
	"""
	# Grab a small slice of the test data
	xs_batch = X_test[:num_samples]
	ys_batch = Y_test[:num_samples]

	# Classifier evaluation must not clamp the label node to the ground truth.
	observation = {"image": xs_batch}
	_, _, y_hat_after, _, _, _ = eval_step(
		model, observation, return_layerwise=False
	)

	# Predictions are returned in a dictionary keyed by node name.
	# Convert the label-node predictions and targets to class integer IDs.
	preds = np.array(jnp.argmax(y_hat_after["label"], axis=-1))
	trues = np.array(jnp.argmax(ys_batch, axis=-1))
	images = np.array(xs_batch)

	# Grid setup
	cols = 5
	rows = int(np.ceil(num_samples / cols))
	fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
	axes = np.array(axes).flatten()

	for i in range(num_samples):
		ax = axes[i]

		# Reshape flattened 784 array back to 28x28 for display
		img = images[i].reshape(28, 28)
		ax.imshow(img, cmap="gray")

		# Color code: Green = Correct, Red = Incorrect
		is_correct = preds[i] == trues[i]
		color = "green" if is_correct else "red"

		ax.set_title(
			f"Pred: {preds[i]} | True: {trues[i]}",
			color=color,
			fontsize=12,
			fontweight="bold",
		)
		ax.axis("off")

	# Turn off empty subplots if num_samples doesn't completely fill the grid
	for i in range(num_samples, len(axes)):
		axes[i].axis("off")

	plt.tight_layout()
	plt.savefig(
		filename, dpi=150, bbox_inches="tight"
	)  # Change filename to .jpg if JPEG is preferred
	plt.close()

	print(f"Evaluation visualization saved to '{filename}'")



# =====================================================================
# 1. Dataset Loader Class
# =====================================================================
class MnistDataloader(object):

  def __init__(
      self,
      training_images_filepath,
      training_labels_filepath,
      test_images_filepath,
      test_labels_filepath,
  ):
    self.training_images_filepath = training_images_filepath
    self.training_labels_filepath = training_labels_filepath
    self.test_images_filepath = test_images_filepath
    self.test_labels_filepath = test_labels_filepath

  def read_images_labels(self, images_filepath, labels_filepath):
    with open(labels_filepath, 'rb') as file:
      magic, size = struct.unpack('>II', file.read(8))
      if magic != 2049:
        raise ValueError(
            f'Magic number mismatch, expected 2049, got {magic}'
        )
      labels = array('B', file.read())

    with open(images_filepath, 'rb') as file:
      magic, size, rows, cols = struct.unpack('>IIII', file.read(16))
      if magic != 2051:
        raise ValueError(
            f'Magic number mismatch, expected 2051, got {magic}'
        )
      image_data = array('B', file.read())

    images = []
    for i in range(size):
      images.append([0] * rows * cols)
    for i in range(size):
      img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
      img = img.reshape(28, 28)
      images[i][:] = img

    return images, labels

  def load_data(self):
    x_train, y_train = self.read_images_labels(
        self.training_images_filepath, self.training_labels_filepath
    )
    x_test, y_test = self.read_images_labels(
        self.test_images_filepath, self.test_labels_filepath
    )
    return (x_train, y_train), (x_test, y_test)


# =====================================================================
# 2. Load and Preprocess for JAX & PCN
# =====================================================================
input_path = 'mnist'
training_images_filepath = join(
    input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte'
)
training_labels_filepath = join(
    input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'
)
test_images_filepath = join(
    input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
)
test_labels_filepath = join(
    input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
)

dataloader = MnistDataloader(
    training_images_filepath,
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath,
)
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = dataloader.load_data()

# Convert images to JAX arrays, normalize to [0, 1], and flatten to (N, 784)
X_train = jnp.array(x_train_raw, dtype=jnp.float32) / 255.0
X_train = jnp.reshape(X_train, (-1, 784))

X_test = jnp.array(x_test_raw, dtype=jnp.float32) / 255.0
X_test = jnp.reshape(X_test, (-1, 784))

# Predictive Coding Networks require continuous targets: convert labels to one-hot vectors (N, 10)
Y_train = jax.nn.one_hot(jnp.array(y_train_raw), num_classes=10)
Y_test = jax.nn.one_hot(jnp.array(y_test_raw), num_classes=10)


# =====================================================================
# 3. Instantiate Runner Functions
# =====================================================================
# Set return_layerwise=True if you want to inspect step-by-step energy traces
RETURN_LAYERWISE = False

activity_optim = optax.adam(learning_rate=5e-3)
param_optim = optax.adamw(learning_rate=6e-5)

train_step = make_train_step(
    param_optim=param_optim,
    activity_optim=activity_optim,
    n_infer_steps=200,
)

eval_step = make_eval_step(activity_optim=activity_optim, n_infer_steps=200)


# Helper function to compute accuracy from target & prediction logits/probabilities
def compute_accuracy(y_hat: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
	preds = jnp.argmax(y_hat, axis=-1)
	targets = jnp.argmax(y_true, axis=-1)
	return jnp.mean(preds == targets)


# =====================================================================
# 4. Test Evaluation Script
# =====================================================================
def run_evaluation(model, X_test, Y_test, batch_size=64):
  """Evaluates model across the entire test set in mini-batches."""
  num_test_batches = len(X_test) // batch_size

  total_energy_before = 0.0
  total_energy_after = 0.0
  total_acc_before = 0.0
  total_acc_after = 0.0

  for i in range(num_test_batches):
    xs_batch = X_test[i * batch_size : (i + 1) * batch_size]
    ys_batch = Y_test[i * batch_size : (i + 1) * batch_size]

    # eval_step returns batched metrics across the mini-batch
    (
        states_curr,
        y_hat_before,
        y_hat_after,
        energy_before,
        energy_after,
        energy_trace,
    ) = eval_step(
        model,
        {"image": xs_batch},  # Do not clamp labels during classifier evaluation.
        return_layerwise=RETURN_LAYERWISE,
    )

    total_energy_before += jnp.mean(energy_before)
    total_energy_after += jnp.mean(energy_after)
    total_acc_before += compute_accuracy(y_hat_before["label"], ys_batch)
    total_acc_after += compute_accuracy(y_hat_after["label"], ys_batch)

  return {
      'energy_before': float(total_energy_before / num_test_batches),
      'energy_after': float(total_energy_after / num_test_batches),
      'acc_before': float(total_acc_before / num_test_batches),
      'acc_after': float(total_acc_after / num_test_batches),
  }


# =====================================================================
# 5. Main Training Loop with Per-Epoch Metrics & Evaluation
# =====================================================================
batch_size = 64
num_batches = len(X_train) // batch_size
num_epochs = 10
key = jax.random.PRNGKey(42)
rng_key, model_key = jax.random.split(key)

def build_specs(num_hidden_layers: int, hidden_size: int = 10, hep_interval=5):
    specs = [
        NodeSpec("image", "input", (784,), clamped=True, release=True),
        NodeSpec("h0", "pcn", (512,), parents=("image",), highway_from_td=("label",)),
    ]

    parent = "h0"

    for i in range(1, num_hidden_layers + 1):
        name = f"h{i}"
        highway_from_td = ()
        highway_from_bu = ()
        if i % hep_interval == 0:
          highway_from_td = ("label",)
          highway_from_bu = ("image",)
        specs.append(
            NodeSpec(name, "pcn", (hidden_size,), parents=(parent,), highway_from_td=highway_from_td, highway_from_bu=highway_from_bu)
        )
        parent = name

    specs.append(
        NodeSpec("label", "pcn", (10,), energy="ce", parents=(parent,), clamped=True)
    )
    return tuple(specs)


specs = build_specs(num_hidden_layers=64, hidden_size=128, hep_interval=5)
config = GraphModelConfig(node_specs=specs, activity_norm="rms", highway_alpha=0.1, bidirectional=True, bu_precision=1e-4, td_act_fn="tanh", bu_act_fn="tanh")
model = GraphModel(config=config, key=model_key)

print(
    f"{'Epoch':<6} | {'Tr Energy (Bef/Aft)':<22} | {'Tr Acc (Bef/Aft)':<18} |"
    f" {'Test Acc (Bef/Aft)':<18}"
)
print('-' * 72)

param_opt_state = param_optim.init(eqx.filter(model, eqx.is_array))

for epoch in range(num_epochs):
	# Shuffle training set every epoch
	rng_key, subkey = jax.random.split(rng_key)
	perm = jax.random.permutation(subkey, len(X_train))
	X_train_shuffled = X_train[perm]
	Y_train_shuffled = Y_train[perm]

	epoch_energy_before = 0.0
	epoch_energy_after = 0.0
	epoch_acc_before = 0.0
	epoch_acc_after = 0.0

	for i in range(num_batches):
		xs_batch = X_train_shuffled[i * batch_size : (i + 1) * batch_size]
		ys_batch = Y_train_shuffled[i * batch_size : (i + 1) * batch_size]

		# Unpack outputs from train_step
		(
			model,
			param_opt_state,
			states_curr,
			y_hat_before,
			y_hat_after,
			energy_before,
			energy_after,
			energy_trace,
		) = train_step(
			model,
			param_opt_state,
			{"image": xs_batch, "label": ys_batch},
			return_layerwise=RETURN_LAYERWISE, init_direction="td",
		)

		# Accumulate training batch metrics from the label-node predictions.
		epoch_energy_before += jnp.mean(energy_before)
		epoch_energy_after += jnp.mean(energy_after)
		epoch_acc_before += compute_accuracy(y_hat_before["label"], ys_batch)
		epoch_acc_after += compute_accuracy(y_hat_after["label"], ys_batch)

	# Compute average training metrics for the epoch
	train_e_before = epoch_energy_before / num_batches
	train_e_after = epoch_energy_after / num_batches
	train_acc_before = epoch_acc_before / num_batches
	train_acc_after = epoch_acc_after / num_batches

	# Evaluate on test set at epoch end
	test_metrics = run_evaluation(model, X_test, Y_test, batch_size=batch_size)

  # Print formatted epoch overview
	print(
		f'{epoch+1:02d}/{num_epochs:02d}  |'
		f' {train_e_before:7.3f} -> {train_e_after:7.3f} |'
		f' {train_acc_before*100:5.1f}% -> {train_acc_after*100:5.1f}% |'
		f' {test_metrics["acc_before"]*100:5.1f}% ->'
		f' {test_metrics["acc_after"]*100:5.1f}%'
	)

# Save 15 samples as a PNG image
save_eval_samples(
    model,
    eval_step,
    X_test,
    Y_test,
    num_samples=15,
    filename="eval_samples.png",
)

# def save_generated_samples(
#     model,
#     n=10,
#     n_steps=200,
#     filename="generated_from_label_mnist.png",
# ):
#     """Generate MNIST images by clamping digit labels and inferring bottom-up."""
#     # Use a longer inference run to allow the generated activities to settle.
#     gen_step = make_eval_step(
#         activity_optim=activity_optim,
#         n_infer_steps=n_steps,
#     )

#     # Generate one image for each digit class, 0 through 9.
#     class_ids = jnp.arange(n) % 10
#     label_onehots = jax.nn.one_hot(class_ids, num_classes=10)

#     labels = jax.nn.one_hot(jnp.arange(10), 10)
#     sp = zero_states(model, 10)
#     init = jax.vmap(lambda s, o: model.init_activities(s, None, o, "bu"))(sp, {"label": labels})
#     sweep_imgs = np.asarray(init["image"]).reshape(-1, 28, 28)   # plot these like the generated ones

#     # Clamp labels, leave the image unspecified, and infer bottom-up.
#     states_curr, *_ = gen_step(
#         model,
#         {"label": label_onehots},
#         return_layerwise=False,
#         init_direction="bu",
#     )

#     # The MNIST image node is flattened to 784 pixels.
#     images = np.asarray(states_curr["image"])
#     images = np.clip(images.reshape(-1, 28, 28), 0.0, 1.0)

#     # Plot generated images.
#     cols = 5
#     rows = int(np.ceil(n / cols))
#     fig, axes = plt.subplots(
#         rows,
#         cols,
#         figsize=(3 * cols, 3 * rows),
#     )
#     axes = np.asarray(axes).flatten()

#     for i, ax in enumerate(axes):
#         ax.axis("off")

#         if i < n:
#             ax.imshow(images[i], cmap="gray", vmin=0.0, vmax=1.0)
#             ax.set_title(f"Class {int(class_ids[i])}", fontsize=11)

#     plt.tight_layout()
#     plt.savefig(filename, dpi=150, bbox_inches="tight")
#     plt.close()

#     print(f"Generated-from-label samples saved to '{filename}'")

from dataclasses import replace


def save_generated_samples(
    model,
    n=10,
    n_steps=200,
    filename="generated_from_label_mnist.png",
    precision_override=None,   # e.g. {"td_precision": 1e-5, "bu_precision": 1.0}; None = use the trained config
):
    """Generate one image per class by clamping the label and leaving the image free.

    Plots two rows so the decoder and the settling can be told apart:
      top row    : the "bu" initialisation sweep ALONE (no settling steps)
      bottom row : the same sweep followed by `n_steps` of settling
    """
    # Optionally swap precisions for generation only. No weight depends on a precision,
    # so the trained nodes can be dropped into a model built with a modified config.
    if precision_override:
        gen_cfg = replace(model.config, **precision_override)
        fresh = GraphModel(gen_cfg, key=jax.random.PRNGKey(0))
        model = eqx.tree_at(lambda m: m.nodes, fresh, model.nodes)

    class_ids = jnp.arange(n) % 10
    label_onehots = jax.nn.one_hot(class_ids, num_classes=10)
    observation = {"label": label_onehots}          # label clamped; "image" omitted -> released, so inferred

    # --- Row 1: the bottom-up initialisation sweep only (no settling) ---
    zero_prev = zero_states(model, n)               # leading batch axis of size n
    init_states = jax.vmap(
        lambda states_prev, obs: model.init_activities(states_prev, None, obs, "bu")
    )(zero_prev, observation)
    sweep_images = np.clip(np.asarray(init_states["image"]).reshape(-1, 28, 28), 0.0, 1.0)

    # --- Row 2: sweep followed by settling ---
    gen_step = make_eval_step(activity_optim=activity_optim, n_infer_steps=n_steps)
    states_curr, *_ = gen_step(model, observation, return_layerwise=False, init_direction="bu")
    settled_images = np.clip(np.asarray(states_curr["image"]).reshape(-1, 28, 28), 0.0, 1.0)

    # --- How distinct are the classes? Mean pairwise correlation between images ---
    def mean_pairwise_corr(images):
        c = np.corrcoef(images.reshape(len(images), -1))
        return float(c[np.triu_indices(len(images), k=1)].mean())

    print(f"mean pairwise correlation across classes | sweep only: {mean_pairwise_corr(sweep_images):.3f}"
          f" | settled: {mean_pairwise_corr(settled_images):.3f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 4.4), squeeze=False)
    for col in range(n):
        for row, imgs in enumerate((sweep_images, settled_images)):
            ax = axes[row, col]
            ax.imshow(imgs[col], cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
        axes[0, col].set_title(f"Class {int(class_ids[col])}", fontsize=10)

    # Row labels (axes are off, so write the text in axes coordinates)
    axes[0, 0].text(-0.1, 0.5, "bu sweep only", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=10)
    axes[1, 0].text(-0.1, 0.5, f"settled ({n_steps} steps)", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Generated-from-label samples saved to '{filename}'")

    return sweep_images, settled_images


save_generated_samples(
    model,
    n=10,
    n_steps=200,
    filename="generated_from_label_mnist.png",
    precision_override={"td_precision": 0.0, "bu_precision": 1.0},
)