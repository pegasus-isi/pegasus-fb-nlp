#!/usr/bin/env python
import os
import pwd
import sys
import time
import glob
import argparse
import logging
from logging import Logger
from Pegasus.DAX3 import *

##################### BEGIN PARAMETER #####################

DATA_PATH	=	"input/"
MONO_PATH	=	DATA_PATH + "mono"
PARA_PATH	=	DATA_PATH + "para"

# If we need to fetch data from server
LANGS		= 	['en', 'fr']
YEARS		= 	[2007]
BASE_URL	=	"http://www.statmt.org/wmt14/training-monolingual-news-crawl/"

# Pre-training parameters
N_MONO		=	10000000  		# number of monolingual sentences for each language
CODES 		=	60000      		# number of BPE codes
N_THREADS 	=	48     			# number of threads in data preprocessing
N_EPOCHS	=	10      		# number of fastText epochs


# # main paths
# UMT_PATH=$PWD
# TOOLS_PATH=$PWD/tools
# DATA_PATH=$PWD/data
# MONO_PATH=$DATA_PATH/mono
# PARA_PATH=$DATA_PATH/para

# # moses
# MOSES=$TOOLS_PATH/mosesdecoder
# TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
# NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
# INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
# REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl

# # fastBPE
# FASTBPE_DIR=$TOOLS_PATH/fastBPE
# FASTBPE=$FASTBPE_DIR/fast

# # fastText
# FASTTEXT_DIR=$TOOLS_PATH/fastText
# FASTTEXT=$FASTTEXT_DIR/fasttext

# # files full paths
# SRC_RAW=$MONO_PATH/all.en
# TGT_RAW=$MONO_PATH/all.fr
# SRC_TOK=$MONO_PATH/all.en.tok
# TGT_TOK=$MONO_PATH/all.fr.tok
# BPE_CODES=$MONO_PATH/bpe_codes
# CONCAT_BPE=$MONO_PATH/all.en-fr.$CODES
# SRC_VOCAB=$MONO_PATH/vocab.en.$CODES
# TGT_VOCAB=$MONO_PATH/vocab.fr.$CODES
# FULL_VOCAB=$MONO_PATH/vocab.en-fr.$CODES
# SRC_VALID=$PARA_PATH/dev/newstest2013-ref.en
# TGT_VALID=$PARA_PATH/dev/newstest2013-ref.fr
# SRC_TEST=$PARA_PATH/dev/newstest2014-fren-src.en
# TGT_TEST=$PARA_PATH/dev/newstest2014-fren-src.fr

###################### END PARAMETER ######################

LOGGER = logging.getLogger(__name__)

"""
Logger configuration
"""

def configure_logger(logger):
	logger.setLevel(logging.INFO)

	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	formatter = logging.Formatter(
		'[%(asctime)s] %(levelname)-8s %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S',
	)
	ch.setFormatter(formatter)
	logger.addHandler(ch)


# The name of the DAX file is the first argument
if len(sys.argv) != 2:
	sys.stderr.write("Usage: %s DAXFILE\n" % (sys.argv[0]))
	sys.exit(1)
daxfile = sys.argv[1]

USER = pwd.getpwuid(os.getuid())[0]

configure_logger(LOGGER)

# Create a abstract dag
LOGGER.info("Creating ADAG...")
dag = ADAG("fb-nlp-nmt")

# Add some workflow-level metadata
dag.metadata("creator", "%s@%s" % (USER, os.uname()[1]))
dag.metadata("created", time.ctime())

wget = []
unzip = []
dataset = []
concat = []
tokenize = []

files_already_there = [os.path.basename(x) for x in glob.glob(MONO_PATH+"/*")]

for lang in range(len(LANGS)):
	wget.append([])
	unzip.append([])
	dataset.append([])

	for year in range(len(YEARS)):
		current_input = File("{0}/news.{1}.{2}.shuffled.gz".format(BASE_URL, YEARS[year], LANGS[lang]))
		input_unziped = current_input.name.split('/')[-1]
		#If the data set is already there
		if input_unziped not in files_already_there:
			LOGGER.info("{} will be downloaded..".format(input_unziped))
			wget[lang].append(Job("wget"))
			wget[lang][year].addArguments("-c", current_input)
			wget[lang][year].uses(current_input, link=Link.INPUT)
			dag.addJob(wget[lang][year])
		else:
			LOGGER.info("{} found in {}".format(input_unziped, MONO_PATH))

		dataset[lang].append(File(input_unziped[:-3]))
		# if the data set is already unzipped
		if input_unziped[:-3] not in files_already_there:
			# gunzip
			unzip[lang].append(Job("gzip"))
			LOGGER.info("{} will be unzipped..".format(input_unziped))
			unzip[lang][year].uses(current_input, link=Link.INPUT)
			unzip[lang][year].uses(dataset[lang][year], link=Link.OUTPUT, transfer=False, register=False)
			dag.addJob(unzip[lang][year])
			unzip[lang][year].addArguments(current_input)
			# Add dependency only of we download the datasets
			if input_unziped not in files_already_there:
				dag.addDependency(Dependency(parent=wget[lang][year], child=unzip[lang][year]))
		else:
			LOGGER.info("{} already unzipped in {}".format(input_unziped, MONO_PATH))

	#####

	## Concatenate -> SRC_RAW=$MONO_PATH/all.lang1

	concat.append(Job("concat"))
	concat[lang].addArguments("file output by all gzip for one language", "N_MONO", "output: $MONO_PATH/all.en")
	lang_raw = File("all.{1}".format(LANGS[lang]))
	concat[lang].uses(all_datasets, link=Link.INPUT)
	concat[lang].uses(lang_raw, link=Link.OUTPUT, transfer=True, register=True)
	dag.addJob(concat1)
	for year in range(len(YEARS)): 
		dag.addDependency(Dependency(parent=unzip[lang][year], child=concat[lang]))

# 	## Tokenize -> SRC_TOK=$MONO_PATH/all.en.tok
# 	tokenize1 = Job("tokenize")
# 	tokenize1.addArguments("file output by all gzip for one language", "N_MONO", "output: $MONO_PATH/all.en")
# 	tokenize1.uses(listing, link=Link.OUTPUT, transfer=True, register=True)
# 	dag.addJob(tokenize1)
# 	dag.addDependency(Dependency(parent=concat1, child=tokenize))

# 	## Tokenize -> SRC_TOK=$MONO_PATH/all.en.tok
# 	tokenize2 = Job("tokenize")
# 	tokenize2.addArguments("file output by all gzip for one language", "N_MONO", "output: $MONO_PATH/all.en")
# 	src_tok = File("{0}/all.{1}.tok".format(MONO_PATH,LANG))
# 	tokenize2.uses(src_tok, link=Link.OUTPUT, transfer=True, register=True)
# 	dag.addJob(tokenize1)
# 	dag.addDependency(Dependency(parent=concat1, child=tokenize))


# ## fastBPE -> $FASTBPE learnbpe $CODES $SRC_TOK $TGT_TOK > $BPE_CODES
# fast_bpe = Job("fastbpe")
# fast_bpe.addArguments("learnbpe", CODES, )
# fast_bpe.uses(listing, link=Link.OUTPUT, transfer=True, register=True)
# dag.addJob(fast_bpe)

# for lang in LANGS:
# 	dag.addDependency(Dependency(parent=concat[lang], child=fast_bpe))



# parser = argparse.ArgumentParser(description="fb-nlp-nmt")
# parser.add_argument("-i", "--input", type=str, nargs="+", help="Input files (if none, datasets will be downloaded)", required=False)
# parser.add_argument("-y", "--years", type=str, nargs="+", help="Years to consider", required=True)
# parser.add_argument("-l", "--langs", type=str, nargs="+", help="Langs to consider (two maximum)", required=True)
# parser.add_argument("-o", "--outdir", type=str, help="Where to output files", required=True)

# parser.add_argument("-t", "--threads", type=int, default=4, help="Number of threads for the training", required=True)
# parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs for the training", required=True)
# parser.add_argument("-m", "--mono", default=10000000, type=int, help="Number of monolingual sentences for each language", required=True)
# parser.add_argument("-b", "--bpe", default=60000, type=int, help="Number of BPE codes", required=True)

# args = parser.parse_args()
# outdir = os.path.abspath(args.indir)
# outdir = os.path.abspath(args.outdir)

# if not os.path.isdir(args.indir):
# 	os.makedirs(indir)
# 	os.makedirs(indir + "/mono")

# if not os.path.isdir(args.outdir):
# 	os.makedirs(outdir)

# Write the DAX to stdout
LOGGER.info("Writing {}".format(daxfile))

with open(daxfile, "w") as f:
	dag.writeXML(f)

LOGGER.info("Done")

