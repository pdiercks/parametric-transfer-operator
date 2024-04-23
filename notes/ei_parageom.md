# Auxiliary problem
The auxiliary problem is a parametric linear elastic problem that is used to
compute the transformation displacement field $\boldsymbol{d}(\boldsymbol{x};\mu)$.
The transformation displacement $\boldsymbol{d}(\boldsymbol{x};\mu)$ can be used to
generate physical meshes by translating the points of the parent (or reference) mesh.

# Weak form subdomain problem
See `weak_form_subdomain_problem.pdf`.

# Generation of physical meshes

1. Define training set (parameter components = number of subdomains)
2. For each $\mu$:
    1. Read parent mesh
    2. For each parameter component ($\mu_i$):
        1. Compute transformation displacement $d(\mu_i)$
        2. Translate domain (mesh points) by $d(\mu_i)$ + $\Delta\,x$, where $\Delta\,x$ is the unit cell length.
3. Merge physical (translated) subdomain meshes to obtain physical oversampling domain.

# EI of Transformation operator

1. Define training set (single domain setting).
2. Compute transformation displacement for each $\mu$.
3. Form transformation operator (UFL expr) and interpolate into quadrature space.
4. Run ei greedy on data from step 3.
