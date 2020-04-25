#!/bin/bash
FILE="$1"
NAME=${FILE%.*}
EXT=${FILE#*.}
pegasus-graphviz $NAME.dax -o $NAME.dot
dot -Tpdf -o img/$NAME.pdf $NAME.dot
dot -Tsvg -o img/$NAME.svg $NAME.dot
dot -Tpng -o img/$NAME.png $NAME.dot

