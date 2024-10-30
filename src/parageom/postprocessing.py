"""Postprocessing module."""

from collections import defaultdict

import numpy as np


def compute_mean_std(dependencies, targets):
    output = {}
    error = defaultdict(list)
    for dep in dependencies:
        data = np.load(dep)
        for key in data.files:
            error[key].append(data[key])

    for key, value in error.items():
        err = np.vstack(value)
        output[f'mean_{key}'] = np.mean(err, axis=0)
        output[f'std_{key}'] = np.std(err, axis=0)
    np.savez(targets[0], **output)
