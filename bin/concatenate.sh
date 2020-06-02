#!/usr/bin/env bash
set -e

# concatenate monolingual data files for one language

while getopts 'm:o:' opt; do
	case $opt in
		m) N_MONO=$OPTARG ;;
		o) SRC_RAW=$OPTARG ;;
	esac
done
shift $(( OPTIND - 1 ))

FILES="$@"

echo "Concatenating monolingual data for: $@"
cat $FILES | head -n $N_MONO > $SRC_RAW
echo "monolingual data concatenated in: $SRC_RAW"

# check number of lines
if ! [[ "$(wc -l < $SRC_RAW)" -eq "$N_MONO" ]]; then 
	echo "ERROR: Number of lines ($(wc -l < $SRC_RAW)) doesn't match! Be sure you have $N_MONO sentences in your monolingual data." 
fi
