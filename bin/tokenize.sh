#!/usr/bin/env bash
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

TOKENIZER=$(pwd)/tokenizer_perl
NORM_PUNC=$(pwd)/normalize-punctuation_perl

while getopts 'i:l:p:o:' opt; do
    case $opt in
        i) INPUT=$OPTARG ;;
        l) LANG=$OPTARG ;;
        p) THREADS=$OPTARG ;;
        o) OUTPUT=$OPTARG ;;
    esac
done

# tokenize data
echo "Tokenize monolingual data for $LANG..."

echo $(pwd)
echo "----"
echo $(ls -alh)

cat $INPUT | $NORM_PUNC -l $LANG | $TOKENIZER -l $LANG -no-escape -threads $THREADS > $OUTPUT

echo "$LANG monolingual data tokenized in: $OUTPUT"
