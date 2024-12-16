"""Postprocessing module."""

from collections import defaultdict

import numpy as np


def compute_mean_std(dependencies, targets):
    output = {}
    error = defaultdict(list)
    # keep track of (max) number of modes
    # differs per realization; cannot stack arrays of different size
    num_modes = defaultdict(list)
    for dep in dependencies:
        data = np.load(dep)
        for key in data.files:
            error[key].append(data[key])
            num_modes[key].append(data[key].size)

    for key, value in error.items():
        n = min(num_modes[key])
        # truncate each array to min number of modes
        truncated = [arr[:n] for arr in value]
        err = np.vstack(truncated)
        output[f'mean_{key}'] = np.mean(err, axis=0)
        output[f'std_{key}'] = np.std(err, axis=0)
    np.savez(targets[0], **output)
