# Parametric transfer operator
This repository contains source code to implement reduced-order models that can be used to simulate geometrically parameterized lattice structures. The reduced-order models are constructed using concepts from localized model order reduction.
A preliminary version of this work is published in the [conference proceedings of the 9th ECCOMAS congress 2024](http://dx.doi.org/10.23967/eccomas.2024.207).
The current version (`main` branch) implements the numerical examples that are described in Chapter 04 of the thesis *Multiscale modeling of mechanical structures via localized model reduction*.
The final version of the thesis can be obtained from [this link](https://research.tue.nl/en/publications/multiscale-modeling-of-mechanical-structures-via-localized-model-/).

## Tree
The following directory structure is used:

* src (any source code)

The `src` directory contains any source code to run the example problems and any
code for plotting or creation of figures.
Each instance of the workflow will be stored locally under `./work` with a
subdirectory for each example.
The outputs (final figures) of the `doit` workflow are stored under `./figures`.

## Compute environment
The compute environment can be instantiated with [`pixi`](https://prefix.dev/).
See `pyproject.toml` and the related `pixi` sections for the definition of _dependencies_ and _tasks_ that may by executed via `pixi run <task>`.

### Steps
1. Download source code for non-conda dependencies `pymor` and `multicode`. Run the following commands each from the root of the project.
```sh
export PYMORSRC=./.pymorsrc # do not change this, see pyproject.toml
git clone -b fenicsx-pd --single-branch git@github.com:pdiercks/pymor.git $PYMORSRC
```
```sh
export MULTISRC=./.multisrc # do not change this, see pyproject.toml
git clone -b main --single-branch git@github.com:pdiercks/multicode.git $MULTISRC
```

2. Install the environment.
```sh
pixi install
```
This will install the _default_ environment.

3. Run the workflow.
```sh
pixi run doit
```
Note that running the complete workflow may take a while.
Alternatively, `pixi shell` will open a shell with the environment activated or any other doit command, e.g. `doit list`, may be executed via `pixi run doit list`.
