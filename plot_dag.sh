#!/bin/bash
FILE="$1"
NAME=${FILE%.*}
EXT=${FILE#*.}
pegasus-graphviz $NAME.dax -o $NAME.dot
dot -Tpdf -o $NAME.pdf $NAME.dot
dot -Tsvg -o $NAME.svg $NAME.dot

