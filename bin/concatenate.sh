#!/usr/bin/env bash
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

# concatenate monolingual data files

FILES="$1"
N_MONO="$2"
SRC_RAW="$3"

echo "Concatenating monolingual data for..."

cat $(ls $FILES | grep -v gz) | head -n $N_MONO > $SRC_RAW

echo "monolingual data concatenated in: $SRC_RAW"

# check number of lines
if ! [[ "$(wc -l < $SRC_RAW)" -eq "$N_MONO" ]]; then 
	echo "ERROR: Number of lines doesn't match! Be sure you have $N_MONO sentences in your EN monolingual data." 
	exit -1
fi

