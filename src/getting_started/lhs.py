"""latin hypercube sampling"""

from typing import Optional
import numpy as np
from pyDOE3 import lhs
from pymor.parameters.base import Mu, Parameters, ParameterSpace


def scale_range(samples: np.ndarray, ranges: np.ndarray):
    n = ranges.shape[0]
    for k in range(n):
        samples[:, k] = ranges[k, 0] + samples[:, k] * (
            ranges[k, 1] - ranges[k, 0]
        )
    return samples


def sample_lhs(parameter_space: ParameterSpace, name: str, **kwargs) -> list[Mu]:
    """Creates a parameter set based on Latin Hypercube design.

    Args:
        parameter_space: The parameter space.
        name: Name of the parameter component to sample.
        kwargs: Optional keyword arguments like `samples`, `criterion` to be passed to `pyDOE3.lhs`.

    Returns:
        The parameter set.

    """
    ndim = parameter_space.parameters[name]
    ranges = np.array([parameter_space.ranges[name]], dtype=np.float32)
    xrange = np.repeat(ranges, ndim, axis=0)
    samples = lhs(ndim, **kwargs)
    scale_range(samples, xrange)

    mus = []
    for s in samples:
        mus.append(parameter_space.parameters.parse(s))

    return mus


if __name__ == "__main__":
    E_range = [1., 2.]
    xr = np.array([E_range], dtype=np.float32)
    ndim = 2
    x = np.repeat(xr, ndim, axis=0)
    num_samples = 50
    criterion = "center"
    samples = lhs(ndim, samples=num_samples, criterion=criterion)
    scale_range(samples, x)

    breakpoint()

    import matplotlib.pyplot as plt

    plt.plot(samples[:, 0], samples[:, 1], "o")
    plt.show()


