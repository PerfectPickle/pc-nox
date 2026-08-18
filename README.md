# pc-nox

This library aims to provide easy to use Equinox (JAX) implementations of cutting edge Predictive Coding (PC) variants, such as  [tPC-H](https://www.biorxiv.org/content/10.64898/2026.07.09.737423v1), [Meta-PCN](https://openreview.net/forum?id=kE5jJUHl9i) , [Highway Error Propagation](https://arxiv.org/abs/2606.22744), and experimental fusion architectures. 

Additionally, convenient and robust checkpoint saving/loading will be implemented, with the goal of supporting modular functionality within larger continual learning meta-architectures.


## Currently Supported Architectures
* **None**: Placeholder.


## Other Features
* **PyHGF Compatability**: Version matches *current* (18/08/26) [pyHGF](https://github.com/ComputationalPsychiatry/pyhgf) shared dependencies for cross compatability. Compatability will be maintained.


## Planned Architectures
* **tPC-H**: Hierarchical temporal Predictive Coding.
* **tPC-E**: Temporal Predictive Coding with eligibility traces.
* **PCN-HEP**: PCN with Highway Error Propagation.
* **Meta-PCN**: Hierarchical temporal Predictive Coding with eligibility traces.


## Planned Features

* **Model Management**: Easy model saving and loading functions.
* **Analytics**: Additional test metrics and data visualisation methods.


## Planned Architecture Experiments
* **tPC-HE**: Hierarchical temporal Predictive Coding with eligibility traces.
* **tPC-H-HEP**: Hierarchical temporal Predictive Coding with highway error propagation.
* **tPC-HE-HEP**: Hierarchical temporal Predictive Coding with eligibility traces and highway error propagation.
* **Meta-tPC-H**: Hierarchical temporal Predictive Coding with Meta-PCN inference dynamics.
* **Meta-tPC-HE**: Hierarchical temporal Predictive Coding with eligibility traces and Meta-PCN inference dynamics.
* **Meta inference + Highway Error Propagation**.
* **Oscillatory PCNs**: Support for oscillatory dynamics.


## Installation

```
# Clone repo
git clone https://github.com/PerfectPickle/pc-nox
cd pc-nox

# Create and activate a Python 3.12 environment
conda create -n pc-nox python=3.12 -y
conda activate pc-nox

# Install the package locally
pip install .

# For CUDA (e.g. 12) usage, upgrade JAX
pip install "jax[cuda12]>=0.4.26,<0.4.32" "jaxlib>=0.4.26,<0.4.32" "numpy>=2.0,<2.5" --force-reinstall

```
