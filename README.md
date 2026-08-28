# pc-nox

This library aims to provide easy to use [Equinox](https://github.com/patrick-kidger/equinox) ([JAX](https://github.com/jax-ml/jax)) implementations of cutting edge Predictive Coding (PC) variants, and experimental fusion architectures.

Additionally, convenient and robust checkpoint saving/loading will be implemented, with the goal of supporting modular functionality within larger continual learning meta-architectures.


## Currently Supported Architectures

* **tPC-H ([Ng-Kee-Kwong et al., 2026](https://www.biorxiv.org/content/10.64898/2026.07.09.737423v1))**: Hierarchical temporal Predictive Coding, with any number of hidden layers, of any size.


## Other Features

* **PyHGF Compatability**: Version matches (18/08/26) [pyHGF](https://github.com/ComputationalPsychiatry/pyhgf) shared dependencies for cross compatability. Compatability will be maintained.
* **Visual Prediction Plotting**: Compare ground truth to pre and post inference predictions in visual environments, with the option to save frames and video. 
* **Flexible Inference Modes**: Supports both step-by-step manual updates for granular control and debugging, as well as end-to-end jax.lax.scan fusion.

## Planned Architectures

* **tPC-E ([Ng-Kee-Kwong et al., 2026](https://www.biorxiv.org/content/10.64898/2026.07.09.737423v1))**: Temporal Predictive Coding with eligibility traces.
* **PCN-HEP ([Mohammadi & Ororbia, 2026](https://arxiv.org/abs/2606.22744))**: PCN with Highway Error Propagation.
* **Meta-PCN ([Ha et al., 2026](https://openreview.net/forum?id=kE5jJUHl9i))**: Hierarchical temporal Predictive Coding with eligibility traces.


## Planned Features

* **Model Management**: Easy model saving and loading functions.
* **Analytics**: Additional test metrics and visualisation methods.
* **Additional Inference Modes**: Inference using ODE solvers.
* **Flexible Training Modes**: Such as scan fused learning.


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


## References

1. **Kidger, P., & Garcia, C. (2021).** *Equinox: Neural networks in JAX via callable PyTrees and filtered transformations*. arXiv. https://doi.org/10.48550/arXiv.2111.00254
2. **Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Katariya, Y., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., & Zhang, Q. (2018).** *JAX: Composable transformations of Python+NumPy programs* (Version 0.3.13) [Computer software]. GitHub. http://github.com/jax-ml/jax
3. **Ng-Kee-Kwong, J., Tang, M., Akam, T., & Bogacz, R. (2026).** *Learning complex temporal dependencies via local synaptic plasticity*. bioRxiv. https://doi.org/10.64898/2026.07.09.737423
4. **PyHGF Development Team. (2026).** *PyHGF: A neural network library for predictive coding* (Version as of August 2026) [Computer software]. GitHub. https://github.com/ComputationalPsychiatry/pyhgf
5. **Mohammadi, A., & Ororbia, A. G. (2026).** *Error highways: Scaling predictive coding to very deep networks*. arXiv. https://doi.org/10.48550/arXiv.2606.22744
6. **Ha, M. H., Kim, H., Sung, Y., Jo, Y., Kang, M. S., & Lee, S. W. (2026).** *Stable and scalable deep predictive coding networks with meta-prediction errors*. International Conference on Learning Representations (ICLR 2026). OpenReview. https://openreview.net/forum?id=kE5jJUHl9i