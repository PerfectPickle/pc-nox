import pickle
from os.path import join
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from pc_nox.engine.runners_static import make_eval_step, make_train_step
from pc_nox.models.pcn.model import PcnModel


# =====================================================================
# 1. Visualization Helper (RGB Images + Class Names)
# =====================================================================
def save_eval_samples(
    model,
    eval_step,
    X_test,
    Y_test,
    fine_names=None,
    num_samples=15,
    filename="eval_samples_cifar100.png",
):
    """Evaluates a batch of test samples, displays RGB images with predicted vs true labels,

    and saves the result to an image file.
    """
    xs_batch = X_test[:num_samples]
    ys_batch = Y_test[:num_samples]

    # Run inference using the eval_step runner
    _, _, y_hat_after, _, _, _ = eval_step(
        model, xs_batch, None, return_layerwise=False
    )

    preds = np.array(jnp.argmax(y_hat_after, axis=-1))
    trues = np.array(jnp.argmax(ys_batch, axis=-1))

    # Reshape flat 3072 arrays back to (32, 32, 3) RGB images for display
    images = np.array(xs_batch).reshape(-1, 32, 32, 3)

    cols = 5
    rows = int(np.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = np.array(axes).flatten()

    for i in range(num_samples):
        ax = axes[i]

        # Clip values to [0, 1] for matplotlib RGB rendering
        img = np.clip(images[i], 0.0, 1.0)
        ax.imshow(img)

        is_correct = preds[i] == trues[i]
        color = "green" if is_correct else "red"

        pred_label = fine_names[preds[i]] if fine_names else f"ID {preds[i]}"
        true_label = fine_names[trues[i]] if fine_names else f"ID {trues[i]}"

        ax.set_title(
            f"Pred: {pred_label}\nTrue: {true_label}",
            color=color,
            fontsize=10,
            fontweight="bold",
        )
        ax.axis("off")

    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Evaluation visualization saved to '{filename}'")


# =====================================================================
# 2. Dataset Loaders for CIFAR-100 Pickled Files
# =====================================================================
def load_cifar100_batch(file_path):
    """Loads a raw CIFAR-100 batch file ('train' or 'test')."""
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")

    images = data_dict[b"data"]
    fine_labels = np.array(data_dict[b"fine_labels"])
    coarse_labels = np.array(data_dict[b"coarse_labels"])

    # Reshape from flat (3072,) -> (3, 32, 32) -> (32, 32, 3)
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return images, fine_labels, coarse_labels


def load_cifar100_meta(meta_path):
    """Loads human-readable fine and coarse class names from the 'meta' file."""
    with open(meta_path, "rb") as f:
        meta_dict = pickle.load(f, encoding="bytes")

    fine_label_names = [name.decode("utf-8") for name in meta_dict[b"fine_label_names"]]
    coarse_label_names = [name.decode("utf-8") for name in meta_dict[b"coarse_label_names"]]
    return fine_label_names, coarse_label_names


# =====================================================================
# 3. Load and Preprocess for JAX & PCN
# =====================================================================
cifar_dir = "cifar-100-python"

train_images, train_fine_labels, _ = load_cifar100_batch(join(cifar_dir, "train"))
test_images, test_fine_labels, _ = load_cifar100_batch(join(cifar_dir, "test"))
fine_names, coarse_names = load_cifar100_meta(join(cifar_dir, "meta"))

# Convert images to JAX arrays, normalize to [0, 1], and flatten to (N, 3072)
X_train = jnp.array(train_images, dtype=jnp.float32) / 255.0
X_train = jnp.reshape(X_train, (-1, 3072))

X_test = jnp.array(test_images, dtype=jnp.float32) / 255.0
X_test = jnp.reshape(X_test, (-1, 3072))

# Convert fine labels to 100-class continuous one-hot vectors (N, 100)
Y_train = jax.nn.one_hot(jnp.array(train_fine_labels), num_classes=100)
Y_test = jax.nn.one_hot(jnp.array(test_fine_labels), num_classes=100)


# =====================================================================
# 4. Instantiate Runner Functions
# =====================================================================
RETURN_LAYERWISE = False

activity_optim = optax.adam(learning_rate=0.01)
param_optim = optax.adam(learning_rate=1e-3)

train_step = make_train_step(
    param_optim=param_optim,
    activity_optim=activity_optim,
    n_infer_steps=20,
)

eval_step = make_eval_step(activity_optim=activity_optim, n_infer_steps=20)


def compute_accuracy(y_hat: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    preds = jnp.argmax(y_hat, axis=-1)
    targets = jnp.argmax(y_true, axis=-1)
    return jnp.mean(preds == targets)


# =====================================================================
# 5. Test Evaluation Script
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

        (
            states_curr,
            y_hat_before,
            y_hat_after,
            energy_before,
            energy_after,
            energy_trace,
        ) = eval_step(model, xs_batch, None, return_layerwise=RETURN_LAYERWISE)

        total_energy_before += jnp.mean(energy_before)
        total_energy_after += jnp.mean(energy_after)
        total_acc_before += compute_accuracy(y_hat_before, ys_batch)
        total_acc_after += compute_accuracy(y_hat_after, ys_batch)

    return {
        "energy_before": float(total_energy_before / num_test_batches),
        "energy_after": float(total_energy_after / num_test_batches),
        "acc_before": float(total_acc_before / num_test_batches),
        "acc_after": float(total_acc_after / num_test_batches),
    }


# =====================================================================
# 6. Main Training Loop with Per-Epoch Metrics & Evaluation
# =====================================================================
batch_size = 64
num_batches = len(X_train) // batch_size
num_epochs = 10

key = jax.random.PRNGKey(42)
rng_key, model_key = jax.random.split(key)

# Input shape: 3072, Hidden: 1024, Output: 100
layer_sizes = [3072, 1024, 100]
act_fns = ["tanh", "identity"] # first layer has no act_fn
model = PcnModel(layer_sizes=layer_sizes, key=model_key, loss="ce", act_fn=act_fns)

print(
    f"{'Epoch':<6} | {'Tr Energy (Bef/Aft)':<22} | {'Tr Acc (Bef/Aft)':<18} |"
    f" {'Test Acc (Bef/Aft)':<18}"
)
print("-" * 72)

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

        epoch_energy_before += jnp.mean(energy_before)
        epoch_energy_after += jnp.mean(energy_after)
        epoch_acc_before += compute_accuracy(y_hat_before, ys_batch)
        epoch_acc_after += compute_accuracy(y_hat_after, ys_batch)

    train_e_before = epoch_energy_before / num_batches
    train_e_after = epoch_energy_after / num_batches
    train_acc_before = epoch_acc_before / num_batches
    train_acc_after = epoch_acc_after / num_batches

    test_metrics = run_evaluation(model, X_test, Y_test, batch_size=batch_size)

    print(
        f"{epoch+1:02d}/{num_epochs:02d}  |"
        f" {train_e_before:7.3f} -> {train_e_after:7.3f} |"
        f" {train_acc_before*100:5.1f}% -> {train_acc_after*100:5.1f}% |"
        f" {test_metrics['acc_before']*100:5.1f}% ->"
        f" {test_metrics['acc_after']*100:5.1f}%"
    )

# Save test samples with class labels as PNG image
save_eval_samples(
    model,
    eval_step,
    X_test,
    Y_test,
    fine_names=fine_names,
    num_samples=15,
    filename="eval_samples_cifar100.png",
)