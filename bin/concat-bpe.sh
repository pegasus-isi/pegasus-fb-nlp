#!/usr/bin/env bash
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

while getopts 'o:' opt; do
	case $opt in
		o) CONCAT_BPE=$OPTARG ;;
	esac
done
shift $(( OPTIND - 1 ))

FILES="$@"

echo "Concatenating source and target monolingual data..."
cat $FILES | shuf > $CONCAT_BPE
echo "Concatenated data in: $CONCAT_BPE"