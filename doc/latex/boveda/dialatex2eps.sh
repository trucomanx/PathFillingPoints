#!/bin/bash


FILENAME=( "Diagrama0" "Diagrama1")

for i in ${FILENAME[*]}; 
do 
    echo XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    echo $i
    echo XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    dia $i.dia -t mp --export=$i.mp
    sed -i 's/\documentclass{minimal}/\documentclass{article}\n\\usepackage{amsmath,amssymb}/' $i.mp

    echo "\documentclass{article}
    \pagestyle{empty}
    \usepackage{graphicx}
    \usepackage{amsmath,amssymb}
    \begin{document}
    \includegraphics[width=\textwidth]{"$i.1"}
    \end{document}" > $i.tex

    mpost  -tex=latex -interaction=nonstopmode $i.mp
    latex $i.tex
    dvips -E -o $i.eps $i

    rm -f $i.log
    rm -f $i.mpx
    rm -f $i.mp
    rm -f $i.1

    rm -f $i.tex
    rm -f $i.aux  
    rm -f $i.log  
    rm -f $i.dvi

done




