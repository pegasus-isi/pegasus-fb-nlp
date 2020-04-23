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
LANGS		=	['en', 'fr']
YEARS		=	[2007,2008]
BASE_URL	=	"http://www.statmt.org/wmt14/training-monolingual-news-crawl/"

# Pre-training parameters
N_MONO		=	10000000		# number of monolingual sentences for each language
CODES 		=	60000			# number of BPE codes
N_THREADS 	=	4				# number of threads in data preprocessing
N_EPOCHS	=	10				# number of fastText epochs


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
src_tok = []
lang_tok = []

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
			wget[lang][year].uses(current_input, link=Link.OUTPUT, transfer=False, register=False)
			dag.addJob(wget[lang][year])
		else:
			LOGGER.info("{0} found in {1}".format(input_unziped, MONO_PATH))

		dataset[lang].append(File(input_unziped[:-3]))

		# if the data set is already unzipped
		if input_unziped[:-3] not in files_already_there:
			# gunzip
			unzip[lang].append(Job("gzip"))

			unzip[lang][year].uses(input_unziped, link=Link.INPUT)
			unzip[lang][year].uses(dataset[lang][year], link=Link.OUTPUT, transfer=False, register=False)
			
			dag.addJob(unzip[lang][year])
			unzip[lang][year].addArguments(input_unziped)

			# Add dependency only of we download the datasets
			if input_unziped not in files_already_there:
				dag.addDependency(Dependency(parent=wget[lang][year], child=unzip[lang][year]))
		else:
			LOGGER.info("{0} already unzipped in {1}".format(input_unziped, MONO_PATH))

	## Concatenate data for each language

	concat.append(Job("concat"))
	lang_raw = File("all.{0}".format(LANGS[lang]))
	concat[lang].addArguments("-m", str(N_MONO), "-o", lang_raw.name, " ".join([x.name for x in dataset[lang]]))
	
	concat[lang].uses(lang_raw, link=Link.OUTPUT, transfer=True, register=True)
	dag.addJob(concat[lang])

	for year in range(len(YEARS)):
		concat[lang].uses(dataset[lang][year], link=Link.INPUT)
		dag.addDependency(Dependency(parent=unzip[lang][year], child=concat[lang]))

	LOGGER.info("{0} monolingual data concatenated in: {1}".format(LANGS[lang], lang_raw.name))

	## Tokenize each language
	tokenize.append(Job("tokenize"))
	lang_tok.append(File("{0}.tok".format(lang_raw.name)))
	tokenize[lang].addArguments("-i", lang_raw.name, "-l", LANGS[lang], "-p", str(N_THREADS), "-o", lang_tok[lang].name)
	
	tokenize[lang].uses(lang_raw, link=Link.INPUT)
	tokenize[lang].uses(lang_tok[lang], link=Link.OUTPUT, transfer=True, register=True)
	
	dag.addJob(tokenize[lang])
	dag.addDependency(Dependency(parent=concat[lang], child=tokenize[lang]))

	LOGGER.info("{0} monolingual data tokenized in: {1}".format(LANGS[lang], lang_tok[lang].name))

## learn BPE codes
fast_bpe = Job("fastbpe")
fast_bpe.addArguments("learnbpe", str(CODES), " ".join([x.name for x in lang_tok]))
bpe_codes = File("bpe_codes")
fast_bpe.setStdout(bpe_codes)
dag.addJob(fast_bpe)

for lang in range(len(LANGS)):
	fast_bpe.uses(lang_tok[lang], link=Link.INPUT)
	dag.addDependency(Dependency(parent=tokenize[lang], child=fast_bpe))

fast_bpe.uses(bpe_codes, link=Link.OUTPUT, transfer=True, register=True)

LOGGER.info("Learning BPE codes")

apply_bpe = []
tok_codes = []

extract_vocab = []
lang_vocab = []

for lang in range(len(LANGS)):
	dag.addDependency(Dependency(parent=concat[lang], child=fast_bpe))

	## Apply BPE codes
	apply_bpe.append(Job("fastbpe"))
	apply_bpe[lang].addArguments("applybpe", str(CODES), " ".join([x.name for x in src_tok]))
	
	tok_codes.append(File("{0}.{1}".format(lang_tok[lang].name, str(CODES))))
	apply_bpe[lang].uses(bpe_codes, link=Link.INPUT)
	apply_bpe[lang].uses(lang_tok[lang], link=Link.INPUT)
	apply_bpe[lang].uses(tok_codes[lang], link=Link.OUTPUT, transfer=True, register=True)
	
	dag.addJob(apply_bpe[lang])
	dag.addDependency(Dependency(parent=fast_bpe, child=apply_bpe[lang]))

	LOGGER.info("BPE codes applied to {0} in: {1}".format(LANGS[lang], tok_codes[lang].name))

	## Extract vocabulary for each language
	extract_vocab.append(Job("fastbpe"))
	extract_vocab[lang].addArguments("getvocab", tok_codes[lang].name)
	lang_vocab.append(File("vocab.{0}.{1}".format(LANGS[lang], str(CODES))))

	extract_vocab[lang].uses(tok_codes[lang], link=Link.INPUT)
	extract_vocab[lang].uses(lang_vocab[lang], link=Link.OUTPUT, transfer=True, register=True)
	extract_vocab[lang].setStdout(lang_vocab[lang])

	dag.addJob(extract_vocab[lang])
	dag.addDependency(Dependency(parent=apply_bpe[lang], child=extract_vocab[lang]))

	LOGGER.info("{0} vocab in: {1}".format(LANGS[lang], lang_vocab[lang].name))


## Extract vocabulary for all languages
extract_vocab_all = Job("fastbpe")
extract_vocab_all.addArguments("getvocab", " ".join([x.name for x in tok_codes]))
lang_vocab_all = File("vocab.{0}.{1}".format("-".join([x for x in LANGS]), str(CODES)))
dag.addJob(extract_vocab_all)
extract_vocab_all.setStdout(lang_vocab_all)

for lang in range(len(LANGS)):
	extract_vocab_all.uses(tok_codes[lang], link=Link.INPUT)
	dag.addDependency(Dependency(parent=apply_bpe[lang], child=extract_vocab_all))

extract_vocab_all.uses(lang_vocab_all, link=Link.OUTPUT, transfer=True, register=True)

LOGGER.info("Full vocab in: {0}".format(lang_vocab_all.name))


## Binarize data
binarize = []
lang_binarized = []

for lang in range(len(LANGS)):
	binarize.append(Job("binarize"))
	binarize[lang].addArguments(lang_vocab_all.name, tok_codes[lang].name)
	dag.addJob(binarize[lang])

	lang_binarized.append(File("{0}.pth".format(tok_codes[lang].name)))

	binarize[lang].uses(lang_vocab_all, link=Link.INPUT)
	binarize[lang].uses(tok_codes[lang], link=Link.INPUT)

	binarize[lang].uses(lang_binarized[lang], link=Link.OUTPUT, transfer=True, register=True)
	binarize[lang].setStdout(lang_binarized[lang])

	dag.addDependency(Dependency(parent=extract_vocab_all, child=binarize[lang]))


	LOGGER.info("{0} binarized data in: {1}".format(LANGS[lang], lang_binarized[lang].name))


### TODO para test
# echo ""
# echo "===== Data summary"
# echo "Monolingual training data:"
# echo "    EN: $SRC_TOK.$CODES.pth"
# echo "    FR: $TGT_TOK.$CODES.pth"
# echo "Parallel validation data:"
# echo "    EN: $SRC_VALID.$CODES.pth"
# echo "    FR: $TGT_VALID.$CODES.pth"
# echo "Parallel test data:"
# echo "    EN: $SRC_TEST.$CODES.pth"
# echo "    FR: $TGT_TEST.$CODES.pth"
# echo ""
### END TODO


###########################################################################
################# Pre-training on concatenated embeddings #################
###########################################################################

## Concatenating source and target monolingual data
concat_bpe = Job("concat-bpe")
lang_bpe_all = File("all.{0}.{1}".format("-".join([x for x in LANGS]), str(CODES)))
concat_bpe.addArguments("-o", lang_bpe_all.name, " ".join([x.name for x in tok_codes]))

concat_bpe.uses(lang_bpe_all, link=Link.OUTPUT, transfer=True, register=True)
dag.addJob(concat_bpe)

for lang in range(len(LANGS)):
	concat_bpe.uses(tok_codes[lang], link=Link.INPUT)
	dag.addDependency(Dependency(parent=apply_bpe[lang], child=concat_bpe))

LOGGER.info("Concatenated shuffled data in: {0}".format(lang_bpe_all.name))

## Actual training with fastText

LOGGER.info("Pre-training fastText on: {0}".format(lang_bpe_all.name))

fasttext = Job("fasttext")
bpe_vec = File("{0}.vec".format(lang_bpe_all.name))
fasttext.addArguments(
	"skipgram", "-epoch", str(N_EPOCHS), 
	"-minCount", "0", "-dim", "512", "-thread", 
	str(N_THREADS), "-ws", "5", "-neg", "10", 
	"-input", lang_bpe_all.name, 
	"-output", lang_bpe_all.name
)

fasttext.uses(lang_bpe_all, link=Link.INPUT)
fasttext.uses(bpe_vec, link=Link.OUTPUT, transfer=True, register=True)
dag.addJob(fasttext)
dag.addDependency(Dependency(parent=concat_bpe, child=fasttext))

LOGGER.info("Cross-lingual embeddings in: {0}".format(bpe_vec.name))


###########################################################################
################################ Training #################################
###########################################################################


training = Job("training")
training_out = File("training.out".format(lang_bpe_all.name))
training.addArguments(TODO)
dag.addJob(training)

training.uses(training_out, link=Link.OUTPUT, transfer=True, register=True)
training.setStdout(training_out)

training.uses(, link=Link.INPUT)
dag.addDependency(Dependency(parent=, child=training))

LOGGER.info("Unsupervised training => ".format(training_out.name))

# python main.py --exp_name test --transformer True --n_enc_layers 4 --n_dec_layers 4 --share_enc 3 --share_dec 3 --share_lang_emb True --share_output_emb True --langs 'en,fr' --n_mono -1 --mono_dataset 'en:./data/mono/all.en.tok.60000.pth,,;fr:./data/mono/all.fr.tok.60000.pth,,' --para_dataset 'en-fr:,./data/para/dev/newstest2013-ref.XX.60000.pth,./data/para/dev/newstest2014-fren-src.XX.60000.pth' --mono_directions 'en,fr' --word_shuffle 3 --word_dropout 0.1 --word_blank 0.2 --pivo_directions 'fr-en-fr,en-fr-en' --pretrained_emb './data/mono/all.en-fr.60000.vec' --pretrained_out True --lambda_xe_mono '0:1,100000:0.1,300000:0' --lambda_xe_otfd 1 --otf_num_processes 30 --otf_sync_params_every 1000 --enc_optimizer adam,lr=0.0001 --epoch_size 500000 --stopping_criterion bleu_en_fr_valid,10


##################### BEGIN EXECUTABLE #####################

### Executable and transformation for tokenize
# toto = Executable(name="toto")

##################### END EXECUTABLE #####################


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

