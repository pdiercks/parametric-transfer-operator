"""Postprocessing module."""

from collections import defaultdict

import numpy as np

# FIXME: Chapter 05
# averaging the singular values is rubbish of course
# Thus, we simply plot the singular values for a single realization with the aim to discuss rapid decay,
# i.e., that HAPOD nicely compresses the data.
# Then, we provide average values for HAPOD (modes / snapshots) in comparison with HRRF
# (number of iterations) in a table to discuss the computational costs for the basis construction.


def parse_logfile(logfile: str, search: dict):
    """Parse `logfile` for patterns in `search`."""
    import re

    def number(string):
        try:
            return int(string)
        except ValueError:
            return float(string)

    rv = defaultdict(list)
    with open(logfile, 'r') as instream:
        for line in instream.readlines():
            for key, chars in search.items():
                if chars in line:
                    filtered = re.sub('[^.0-9e-]', '', line.split(chars)[1])
                    rv[key].append(number(filtered.strip('.')))
    return rv


def average_hapod_data(example):
    from pymor.core.pickle import load

    # Extract HAPOD data (num_modes / num_snapshots)
    # and take the average over realizations for each transfer problem
    keys = ['num_snapshots', 'num_modes']
    average = defaultdict(list)  # list of average values for each key
    for j in range(11):
        local_values = {}
        for k in keys:
            local_values[k] = defaultdict(list)
            for nreal in range(example.num_real):
                with example.hapod_summary(nreal, j).open('rb') as fh:
                    data = load(fh)
                    local_values[k][j].append(data[k])
            assert len(local_values[k][j]) == example.num_real
            average[k].append(np.average(local_values[k][j]))
    for k in keys:
        assert len(average[k]) == 11
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
