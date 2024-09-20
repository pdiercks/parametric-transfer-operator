"""Test discrepancy of LHS design."""

import numpy as np
from pyDOE3 import lhs
from scipy.stats import qmc

# from parageom.lhs import scale_range

d = 4  # dimension of the hypercube [0, 1)^d
n = 200  # number of samples
random_seed = 1452222

l_bounds = [0.0, 0.0]
u_bounds = [5.0, 5.0]

sampler = qmc.LatinHypercube(d)
samples_scipy = sampler.random(n)
# scaled_scipy = qmc.scale(samples_scipy, l_bounds, u_bounds)

current = lhs(d, samples=n, criterion='center')
# scaled_current = scale_range(current, np.vstack([l_bounds, u_bounds]).T)

# ds = qmc.discrepancy(samples_scipy)
# dc = qmc.discrepancy(current)
# breakpoint()

disc_pydoe = {}
criterions = {
    'random': None,
    'center': 'center',
    'maximin': 'maximin',
    'centermaximin': 'centermaximin',
    'correlation': 'correlation',
}
for key, value in criterions.items():
    lhd = lhs(d, samples=n, criterion=value, random_state=random_seed)
    disc_pydoe[key] = qmc.discrepancy(lhd)


options = {
    'optimization': [None, 'random-cd', 'lloyd'],
}
disc_scipy = {}
for opt in options['optimization']:
    sampler = qmc.LatinHypercube(d, optimization=opt, seed=random_seed)
    disc = qmc.discrepancy(sampler.random(n))
    disc_scipy[opt] = disc

print(disc_pydoe)
print(disc_scipy)

ind_min_pydoe = np.argmin(list(disc_pydoe.values()))
criterion_min_pydoe = list(disc_pydoe.keys())[ind_min_pydoe]
value_min_pydoe = list(disc_pydoe.values())[ind_min_pydoe]

ind_min_scipy = np.argmin(list(disc_scipy.values()))
criterion_min_scipy = list(disc_scipy.keys())[ind_min_scipy]
value_min_scipy = list(disc_scipy.values())[ind_min_scipy]

print(f'min scipy with {criterion_min_scipy} is {value_min_scipy}.')
print(f'min pydoe with {criterion_min_pydoe} is {value_min_pydoe}.')

# the results of pydoe are not consistent?
# If I change the random seed, the discrepancy value changes (expected), but also
# the best criterion changes.
# For scipy, I consistently get result that 'random-cd' has smallest discrepancy value.

# Moreover, discrepancy of scipy lhs is smaller than pydoe design.
breakpoint()
