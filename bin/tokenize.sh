#!/usr/bin/env bash
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

TOKENIZER=tokenizer.perl
NORM_PUNC=normalize-punctuation.perl

CONCAT_DATA="$1"
LANG="$2" # en, fr, de etc
N_THREADS="$3"
OUTPUT="$4"

# tokenize data
echo "Tokenize monolingual data for $LANG..."

cat "$CONCAT_DATA" | $NORM_PUNC -l $LANG | $TOKENIZER -l $LANG -no-escape -threads $N_THREADS > $OUTPUT

echo "$LANG monolingual data tokenized in: $OUTPUT"
