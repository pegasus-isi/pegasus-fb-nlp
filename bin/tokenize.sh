#!/usr/bin/env bash

set -e

TOKENIZER=/UnsupervisedMT/NMT/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl
NORM_PUNC=/UnsupervisedMT/NMT/tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl

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

cat $INPUT | $NORM_PUNC -l $LANG | $TOKENIZER -l $LANG -no-escape -threads $THREADS > $OUTPUT

echo "$LANG monolingual data tokenized in: $OUTPUT"
