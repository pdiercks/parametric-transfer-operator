"""Postprocessing module."""

import numpy as np


def compute_mean_std(keywords, dependencies, targets):
    output = {}
    for key in keywords:
        error = []
        for dep in dependencies:
            data = np.load(dep)
            error.append(data[key])
        error = np.vstack(error)
        output[f'mean_{key}'] = np.mean(error, axis=0)
        output[f'std_{key}'] = np.std(error, axis=0)
    np.savez(targets[0], **output)
