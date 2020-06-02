#!/usr/bin/env bash

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