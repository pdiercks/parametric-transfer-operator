"""Postprocessing module."""

from collections import defaultdict

import numpy as np

# FIXME
# averaging the singular values is rubbish of course
# Thus, we simply plot the singular values for a single realization with the aim to discuss rapid decay, i.e., that HAPOD nicely compresses the data.
# Then, we provide average values for HAPOD (modes / snapshots) in comparison with HRRF (number of iterations) in a table to discuss the computational costs for the basis construction.

# TODO For the HRRF data I have to parse the logfiles.
def parse_hrrf_logs():
    pass

def average_hapod_data(example):
    from pymor.core.pickle import load
    # Extract HAPOD data (num_modes / num_snapshots)
    # and take the average over realizations for each transfer problem
    keys = ['num_snapshots', 'num_modes']
    average = defaultdict(list)
    for j in range(11):
        local_values = defaultdict(list)
        for nreal in range(example.num_real):
            with example.hapod_summary(nreal, j).open('rb') as fh:
                data = load(fh)
                for k in keys:
                    local_values[k].append(data[k])
        for k in keys:
            assert local_values[k].size == example.num_real
            average[k] = np.average(local_values[k])
    return average


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
