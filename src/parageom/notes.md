# Generation of physical meshes

1. Define training set (parameter components = number of subdomains)
2. For each mu:
    1. Read parent mesh
    2. For each parameter component (mu_i):
        1. Compute transformation displacement d(mu_i)
        2. Translate domain (mesh points) by d(mu_i) + Δx, where Δx is the unit cell length.
3. Merge physical (translated) subdomain meshes to obtain physical oversampling domain.

# Auxiliary problem

- [ ] add parameters as class member such that auxiliary problem instance can check for correctness of parameter value
