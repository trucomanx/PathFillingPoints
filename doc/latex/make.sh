#!/bin/bash


pdflatex -synctex=1 -interaction=nonstopmode main.tex
biber main
pdflatex -synctex=1 -interaction=nonstopmode main.tex
pdflatex -synctex=1 -interaction=nonstopmode main.tex
pdflatex -synctex=1 -interaction=nonstopmode main.tex

./clean.sh
