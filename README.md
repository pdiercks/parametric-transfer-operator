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
For the simulations an apptainer container is used.
Thus, apptainer needs to be installed on the system.

```sh
mamba install --channel conda-forge apptainer
```

### Steps to build the container

#### Development

First create container in sandbox format.
```sh
apptainer build --sandbox muto-env/ docker://dolfinx/dolfinx:nightly
```
Download source code for additional dependencies `multicode` and `pymor`.
```sh
git clone git@github.com:pdiercks/pymor.git PYMORSRC && cd PYMORSRC && git checkout feniscx-pd
git clone git@github.com:pdiercks/multicode.git MULTISRC && cd MULTISRC && git checkout v0.8.0
```
Note, that at the moment the multicode repo at github is private. Replace with the bam server url.
Also, PYMORSRC and MULTISRC should be placed under `$HOME`. Otherwise, the location of the source files needs to be explicitly bind mounted into the container.
Next, enter the container and install dependencies in addition to fenicsx.
```sh
apptainer shell --fakeroot --writable muto-env/
```
Inside the container run:
```sh
apt-get -qq update && \
apt-get -y install imagemagick libgl1-mesa-glx xvfb && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*
python3 -m pip install h5py meshio sympy doit pyDOE3 coverage
python3 -m pip install --no-cache-dir pyvista==0.43.3
python3 -m pip install --editable PYMORSRC
python3 -m pip install --editable MULTISRC
```
Optionally check editable install was successfull:
```sh
ls /usr/local/lib/python3.10/dist-packages/ | grep ".pth"
```
Set environment variables, in particular
modify PATH to be able to use `$HOME/texlive` install:
```sh
export PATH=~/texlive/bin/x86_64-linux:$PATH
export LC_ALL=C
```

#### Production

Something like
```sh
apptainer build --build-arg TAG=stable production.sif muto-env.def
```
Note, that usage of `muto-env.def` is not tested.
Note, that the last command in the `%post` section does not work, because
the multicode repository is currently private on github.
Note, that `--build-arg TAG=stable` will pull the latest stable release
of the dolfinx docker image.

##### ToDo

- [ ] make multicode a public repository
- [ ] extend muto-env.def for production; install texlive
- [ ] add section `test` to muto-env.def
- [ ] add section `runscript` to muto-env.def?
