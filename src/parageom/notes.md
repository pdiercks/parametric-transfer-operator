# Generation of physical meshes

1. Define training set (parameter components = number of subdomains)
2. For each $\mu$:
    1. Read parent mesh
    2. For each parameter component ($\mu_i$):
        1. Compute transformation displacement $d(\mu_i)$
        2. Translate domain (mesh points) by $d(\mu_i)$ + $\Delta\,x$, where $\Delta\,x$ is the unit cell length.
3. Merge physical (translated) subdomain meshes to obtain physical oversampling domain.

# Auxiliary problem

- [x] add parameters as class member such that auxiliary problem instance can check for correctness of parameter value

# EI of Transformation operator

1. Define training set (single domain setting).
2. Compute transformation displacement for each $\mu$.
3. Form transformation operator (UFL expr) and interpolate into quadrature space or DG-0 space?

The quadrature spaces are broken for rank > 0 tensors. Need to wait for v0.8.0 or use docker.
If the function is element of DG0 space, can I simply put it in the weak form or will that lead to errors since FE spaces are different?

4. Run ei greedy on data from step 3.
