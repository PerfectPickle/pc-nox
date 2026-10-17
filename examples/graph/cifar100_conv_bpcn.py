import pickle
from os.path import join
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

# CHANGED: GraphModel + the combined batch/time runners instead of PcnModel + runners_static.
# (Adjust the package path to wherever you placed the graph package.)
from pc_nox.models.graph import GraphModel, GraphModelConfig, NodeSpec
from pc_nox.models.graph.runners import make_eval_step, make_train_step
from pc_nox.models.graph.optim_utils import grouped_optimiser

from pc_nox.models.graph.runners import zero_states  

def diagnose(model, tag, n=64):
    rms = lambda a: float(jnp.sqrt(jnp.mean(a ** 2)))
    xs, ys = X_test[:n], Y_test[:n]
    obs = {"image": xs, "label": ys}                   # label clamped, so settling has a driver

    # the starting point z0 (the feed-forward sweep)
    s0 = jax.vmap(lambda sp, o: model.init_activities(sp, None, o))(zero_states(model, n), obs)
    # the settled states, plus the feed-forward logits and energies
    sT, y_before, _, e_before, e_after, _ = eval_step(model, obs)

    print(f"--- {tag} ---")
    for k in sT:
        print(f"{k:7s} z0 rms {rms(s0[k]):8.4f} | settled move {rms(sT[k] - s0[k]):8.4f}")
    print(f"logit std {float(y_before['label'].std()):.4f} | "
          f"energy_before {float(jnp.mean(e_before)):.3f} (CE at init ~ ln 100 = 4.6)")

    for name in ("conv1", "conv2", "conv3", "fc", "label"):
        w = model.nodes[name].op_td.weight
        print(f"{name:6s} mean/std = {float(w.mean() / w.std()):+.3f}")

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

    # CHANGED: only the image is observed at test time. The label node is `release`d, so omitting it makes it a
    # free state that settles together with the hidden nodes; its settled activity is the classification.
    #states_curr, _, _, _, _, _ = eval_step(model, {"image": xs_batch}, return_layerwise=False)
    # In save_eval_samples():
    _, _, y_hat_after, _, _, _ = eval_step(model, {"image": xs_batch}, return_layerwise=False)

    preds = np.array(jnp.argmax(y_hat_after["label"], axis=-1))  # Changed from states_curr["label"]

    trues = np.array(jnp.argmax(ys_batch, axis=-1))

    # CHANGED: images are (N, 3, 32, 32) channels-first now; matplotlib wants (32, 32, 3)
    images = np.array(xs_batch).transpose(0, 2, 3, 1)

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

# CHANGED: keep the spatial structure for the conv nodes. Normalise to [0, 1] and go
# (N, 32, 32, 3) -> (N, 3, 32, 32) channels-first, instead of flattening to (N, 3072).
X_train = jnp.transpose(jnp.array(train_images, dtype=jnp.float32) / 255.0, (0, 3, 1, 2))
X_test = jnp.transpose(jnp.array(test_images, dtype=jnp.float32) / 255.0, (0, 3, 1, 2))

# Convert fine labels to 100-class continuous one-hot vectors (N, 100)
Y_train = jax.nn.one_hot(jnp.array(train_fine_labels), num_classes=100)
Y_test = jax.nn.one_hot(jnp.array(test_fine_labels), num_classes=100)


# =====================================================================
# 4. Instantiate Runner Functions
# =====================================================================

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
        ) = eval_step(model, {"image": xs_batch}, return_layerwise=RETURN_LAYERWISE)   # CHANGED: label omitted

        total_energy_before += jnp.mean(energy_before)
        total_energy_after += jnp.mean(energy_after)
        total_acc_before += compute_accuracy(y_hat_before["label"], ys_batch)
        total_acc_after += compute_accuracy(y_hat_after["label"], ys_batch)

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

# CHANGED: BIDIRECTIONAL PREDICTIVE CODING (Oliviers, Tang & Bogacz, ICLR 2026) -- the bidirectional twin of
# cifar100_conv_pcn.py. Same straight line, data, and optimisers; every node that has children also predicts itself
# from them:
#
#   image <-> c1 <-> c2 <-> fc <-> label
#
# Vocabulary, because it trips everyone up: in this graph the image is the ROOT, so a node's `td` prediction (from its
# parent) is the paper's DISCRIMINATIVE direction (image -> label, weights V) and its `bu` prediction (from its child) is
# the paper's GENERATIVE direction (label -> image, weights W). The paper's "bottom-up amortised initialisation sweep"
# is therefore our default `td` sweep.
#
#   * `bidirectional=True` (model-wide): every node WITH CHILDREN gets a bu path. The image is a data root, so it
#     gets a bu-ONLY conv node (the paper's generative term on x1); the label is a leaf, so it stays td-only.
#   * `release=True` on image and label: when omitted from `observation` they become free states. Classification =
#     image observed, label free; generation = label observed, image free.
#   * td_precision / bu_precision are the paper's alpha_disc / alpha_gen. The paper finds the generative weight must be
#     MUCH smaller than the discriminative one (1e-3 .. 1e-8): top-down errors are orders of magnitude larger, and with
#     equal weights they swamp the classification signal. 1e-3 is a starting point, not a tuned value.
#   * The label uses squared error against the one-hot vector (the paper's energy), not cross-entropy.
config = GraphModelConfig(
    node_specs=(
        # --- INPUT ---
        NodeSpec(
            "image",
            "input",
            (3, 32, 32),
            clamped=True,
            release=False,
            bu_act_fn="identity",
        ),
        # --- STAGE 1: (64 channels, 32x32) ---
        NodeSpec(
            "conv1",
            "pcn_conv",
            (64, 32, 32),
            parents=("image",),
            td_act_fn="gelu",
            options={"td_stride": 1, "bu_stride": 2, "bu_output_padding": 1},
        ),
        # --- STAGE 2: Downsample to (128 channels, 16x16) ---
        NodeSpec(
            "conv2",
            "pcn_conv",
            (128, 16, 16),
            parents=("conv1",),
            td_act_fn="gelu",
            options={"td_stride": 2, "bu_stride": 2, "bu_output_padding": 1},
        ),
        # --- STAGE 3: Downsample to (256 channels, 8x8) ---
        NodeSpec(
            "conv3",
            "pcn_conv",
            (256, 8, 8),
            parents=("conv2",),
            td_act_fn="gelu",
            options={"td_stride": 2, "bu_stride": 1},
        ),
        # --- DENSE TAIL ---
        NodeSpec("fc", "pcn", 1024, parents=("conv3",), td_act_fn="tanh",),
        NodeSpec(
            "label",
            "pcn",
            100,
            parents=("fc",),
            td_act_fn="identity",
            clamped=True,
            release=False,
            energy="meta_ce"
        ),
    ),
    bidirectional=False,
    td_precision=1.0,
    bu_precision=1.0,
    energy="meta",
    weight_norm="variance",
)
model = GraphModel(config, key=model_key)


RETURN_LAYERWISE = False

# CHANGED: learning rates. The conv graph is far more sensitive to these than the dense PcnModel was. Measured on
# CIFAR-100 with this graph: param lr 1e-3 collapsed the feedforward net to chance in every run (with activity lr 0.01
# or 0.001); param lr 1e-4 trained, and lowering the activity lr to 0.001 as well gave the best result (~16.6% test
# after 10 epochs, still rising). Tune these first if you change the architecture.
ACTIVITY_LR = 0.2
#activity_optim = grouped_optimiser(config, optax.sgd, ACTIVITY_LR, {"kind:pcn_conv": 0.1})
#param_optim = optax.adam(learning_rate=0.001)
activity_optim = optax.sgd(learning_rate=ACTIVITY_LR)
PARAM_LR = 0.0003
param_optim = grouped_optimiser(config, optax.adamw, PARAM_LR, {"kind:pcn": 0.3})
# param_optim = optax.adam(learning_rate=0.0001)

train_step = make_train_step(
    param_optim=param_optim,
    activity_optim=activity_optim,
    n_infer_steps=20,
)

eval_step = make_eval_step(activity_optim=activity_optim, n_infer_steps=20)


print(
    f"{'Epoch':<6} | {'Tr Energy (Bef/Aft)':<22} | {'Tr Acc (Bef/Aft)':<18} |"
    f" {'Test Acc (Bef/Aft)':<18}"
)
print("-" * 72)

param_opt_state = param_optim.init(eqx.filter(model, eqx.is_array))

diagnose(model, "step 0 (untrained)")

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
            {"image": xs_batch, "label": ys_batch},   # CHANGED: every clamped node is one entry of the observation dict
            return_layerwise=RETURN_LAYERWISE,
        )

        epoch_energy_before += jnp.mean(energy_before)
        epoch_energy_after += jnp.mean(energy_after)
        epoch_acc_before += compute_accuracy(y_hat_before["label"], ys_batch)
        epoch_acc_after += compute_accuracy(y_hat_after["label"], ys_batch)

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

    if epoch == 0:
        diagnose(model, "after epoch 1")

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


# CHANGED: the reverse query. Clamp ONLY the label; the image is released, initialised by the generative (bu) sweep
# from the label up, then settled. More inference steps than in training, as in the paper.
def save_generated_samples(model, fine_names=None, n=10, n_steps=100, filename="generated_from_label_cifar100.png"):
    gen_step = make_eval_step(activity_optim=activity_optim, n_infer_steps=n_steps)
    onehots = jax.nn.one_hot(jnp.arange(n), 100)
    states_curr, *_ = gen_step(model, {"label": onehots}, init_direction="bu")
    images = np.clip(np.array(states_curr["image"]).transpose(0, 2, 3, 1), 0.0, 1.0)

    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for i, ax in enumerate(np.array(axes).flatten()):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i])
            ax.set_title(fine_names[i] if fine_names else f"class {i}", fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Generated-from-label samples saved to '{filename}'")


# save_generated_samples(model, fine_names=fine_names)
