# 2023_11_mu_to
The `mu` stands for *parametric* and `to` stands for *transfer operator*.
This is work in progress for a paper on the extension of the work in [this paper](https://git.bam.de/mechanics/paper/2020_02_multiscale) to the parametric setting.

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
