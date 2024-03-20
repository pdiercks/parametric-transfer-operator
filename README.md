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
