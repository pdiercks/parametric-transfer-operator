# 2023_11_mu_to
The `mu` stands for *parametric* and `to` stands for *transfer operator*.
This is work in progress for the extension of the work in [this paper](https://git.bam.de/mechanics/paper/2020_02_multiscale) to the parametric setting.
At the moment (March 2024) a conference proceedings paper (short term, ECCOMAS June 2024) and a journal paper (long term) are planned.

## Conference proceedings
Initially, the aim was to study different training strategies decoupled from the
(complex) implementation of parametrized geometries.
However, the comparison of the training strategies was completely moved to the journal
paper to avoid any overlap.

* Focus on application: variation of void radius

## Journal paper
The preliminary version of a comparison of the training strategies (March 2024)
is archived using a git tag _heuristic-rrf_.

* Focus on comparison of training strategies _HAPOD_ and _Heuristic randomized range finder_
* If possible, extend the proof in BS2018 to parametric transfer operators

## Tree
The following directory structure is used:

* src (any source code)
* system (system information)
* figures (.pdf, .png)
* tables (.pdf)
* paper (latex sources and pdf)
* journal (final version of the pdf)
* notes (.md, .tex)

The `src` directory contains any source code to run the example problems and any
code for plotting or creation of figures.
Each instance of the workflow will be stored locally under `./work` with a
subdirectory for each example. The output of the `doit` workflow is `paper/paper.pdf`.
Each figure and table included in the paper is stored separately in `figures` and `tables`.
The `notes` dir is used to write down ideas and important equations (project sheet).

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
git clone -b main --single-branch https://git.bam.de/mechanics/pdiercks/multicode.git $MULTISRC
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
Alternatively, `pixi shell` will open a shell with the environment activated or any other doit command, e.g. `doit list`, may be executed via `pixi run doit list`.
