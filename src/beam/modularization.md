# Modularization muto
Break down certain parts of the workflow into black boxes with input and output.

## Identified features that would be needed

- import of _modules_ (e.g. python files) via `importlib` (import of methods such as _discretize_fom_ and _discretize_oversapmling_problem_)
- _ParametricTransferProblem_ with method `.update(mu)` implemented by user in _discretize_oversampling_problem_

## Extension

### Input
- ExampleData (data class)
- configuration/info of which data (functions) is extended into which subdomain: currently `beam.cell_to_config` and I have one file for each configuration; number of configurations depends on geometry and fine scale subdomain grids used. If number of configurations is smaller than number of cells, one could always duplicate the data quite easily to always guarantee that there is exactly one file for each coarse grid cell. This intermediate step could then also deal with ownership (see lines 105-117). 
- MultiscaleProblemDefinition <-- coarse grid & fine scale grid

### Output
- .npz file for each subdomain

## Run local ROM

### Input
- DataClass to get mesh etc.
- MultiscaleProblemDefinition
- Discretized FOM (some function that is imported; would need to write this to disk, but FEniCS objects are not pickleable)

### Important operations
- BasesLoader.read_bases(); same for each problem as long as subdomain has quadrilateral shape
- reconstruction: requires dofmap and bases, but is the same for each problem
- assemble_system: currently only homogeneous Dirichlet BCs are implemented (can be generalized easily); COOMatrixOperator may not be usable for different parametrizations (very likely)

### Output
- ROM error against number of modes

## Range finder (construction of fine scale edge basis functions)

### Input
- DataClass
- MultiscaleProblemDefinition
- function to discretize oversampling problem (highly dependent on parametrization)
- ability to update discretized oversampling problem to new parameter value (e.g. update material; dependent on parametrization)
- neumann problem (neumann if not None, could be defined via MultiscaleProblemDefinition, neumann problem = transfer problem; see 3rd bullet point)

#### ParametricTransferProblem
If the transfer problem has the functionality to update to a new parameter value (implemented by the user for whatever transfer problem they are solving), then the range finder script could simply call this method.
In the current setting `transfer_problem.update(mu)` would be the same as `transfer_problem.update_material`.
However, at the moment I am not sure whether this approach is going to work with any parametrization and in particular
parametrized geometries...

Still the function `discretize_oversampling_problem` is highly problem dependent and how would one pass
this method to the script/module `heuristic_range_approx.py`?
I could use the same approach as in the fenicsx constitutive paper.

### Output
- .npz file for each configuration
