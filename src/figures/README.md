# Figures
Source code for standalone figures (tikzpictures, pgfplots, etc.).

## ToDo
Add tasks to compile standalone figures like this
```sh
latexmk -pdf -cd -out2dir=/home/pdiercks/projects/muto/paper/img ./src/figures/*.tex
```

In `paper.tex`, include the pdf files under `./paper/img`.
Maybe, for the proceedings this is not really necessary, but including the standalone tex file directly does not work if it's not a relative path under root.
If the result is not satisfactory, then it would be better to change the directory structure though.
