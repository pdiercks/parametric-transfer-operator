# Localized MOR

Things to consider building the localized ROM.

## Oversampling

* If the coarse grid is 1x10 then a single oversampling problem should be enough.
* The result should be an orthonormal basis for the unit cell.
* Add single solution accounting for the neumann data to the basis.
* This basis is then decomposed into coarse and fine scale parts; and extended into the subdomain.
* Finally, projection of the stiffness matrix of the unit cell to obtain the local reduced operator (i.e. `data`, see below)

## Assembly of the global operator

* The global operator is parametric.
* Assembly can only be carried out after mu is known.
* Implement `SparseCOOMatrixOperator` or just `COOMatrixOperator`

Procedure:
1. Loop over coarse grid cells to gather dof indices (`rows`, `cols`).
2. `rows` & `cols` is passed as input argument to `COOMatrixOperator` together with `data`.
3. `COOMatrixOperator.assemble(mu)` builds the global matrix.
4. At this point usable as usual matrix operator.

Notes:
* The loop over the coarse grid is only done once.
* `data` is the global array, holding all local operators with _BCs_ applied.
* The `assemble(mu)` method needs to figure out how to multiply `data` with `mu` correctly.
* If $Q_a>1$ on subdomain level, I would need a linear combination of `COOMatrixOperator`s to form the global operator.

### Sparsity Pattern
Assume same sparsity pattern for all matrices.
Consequently for any entry that would hold an entry not equal to zero if no BCs were applied one should actively write a zero value. Otherwise the sparsity pattern is changed. Not sure if this will lead to problems or less efficient code if linear combinations of those sparse matrices are formed later on.

### Apply boundary conditions to COOMatrixOperator
Issue: with COOMatrixOperator the loop over the cells is only carried out once to determine `rows` and `cols`. However, for the copy-local-to-global approach the local element matrix (lhs) and vector (rhs) need to be available inside the loop. --> Thus, I have to prepare the global `data` array (for both the lhs and rhs).

see `test_coo_assembly.py`.
As usual for the global operator we would then need the matrices arising from discretization of each `a_q`, but also a single matrix to apply BCs.
