from array import array
from os.path import join
import struct
import jax
import jax.numpy as jnp
import numpy as np
from pc_nox.models.runners_static import make_eval_step, make_train_step
from pc_nox.models.pcn.model import PcnModel
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

	# Run inference using your existing eval_step runner
	_, _, y_hat_after, _, _, _ = eval_step(
		model, xs_batch, ys_batch, return_layerwise=False
	)

	# Convert predicted probabilities/logits and targets to class integer IDs
	preds = np.array(jnp.argmax(y_hat_after, axis=-1))
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

activity_optim = optax.sgd(learning_rate=0.01)
param_optim = optax.adam(learning_rate=1e-3)

train_step = make_train_step(
    param_optim=param_optim,
    activity_optim=activity_optim,
    n_infer_steps=20,
)

eval_step = make_eval_step(activity_optim=activity_optim, n_infer_steps=20)


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
    ) = eval_step(model, xs_batch, ys_batch, return_layerwise=RETURN_LAYERWISE)

    total_energy_before += jnp.mean(energy_before)
    total_energy_after += jnp.mean(energy_after)
    total_acc_before += compute_accuracy(y_hat_before, ys_batch)
    total_acc_after += compute_accuracy(y_hat_after, ys_batch)

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


model = PcnModel(layer_sizes=[784, 512, 10], key=model_key)

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
			xs_batch,
			ys_batch,
			return_layerwise=RETURN_LAYERWISE,
		)

		# Accumulate training batch metrics
		epoch_energy_before += jnp.mean(energy_before)
		epoch_energy_after += jnp.mean(energy_after)
		epoch_acc_before += compute_accuracy(y_hat_before, ys_batch)
		epoch_acc_after += compute_accuracy(y_hat_after, ys_batch)

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