#!/usr/bin/env python

import os
import pwd
import sys
import time
import glob
import argparse
import logging
import traceback
from logging import Logger

from Pegasus.DAX3 import *

######################## WORKFLOW PARAMETER ########################

DAG_ID					= "fb-nlp-nmt"
DATA_PATH				= "input/"

########################### END WORKFLOW ###########################

######################## PREPROCESS PARAMETER ######################

MONO_PATH				= DATA_PATH + "mono"
PARA_PATH				= DATA_PATH + "para"

LANGS					= ['en', 'fr']
YEARS					= [2007, 2008]

# If we need to fetch data from server
BASE_URL				= "http://www.statmt.org/wmt14/training-monolingual-news-crawl/"

########################## END PREPROCESS #########################

####################### PRETRAINING PARAMETER #####################

N_MONO					= 10000000			# number of monolingual sentences for each language
CODES 					= 60000				# number of BPE codes
N_THREADS 				= 4					# number of threads in data preprocessing
N_EPOCHS				= 10				# number of fastText epochs

########################## END PRETRAINING #######################

######################### TRAINING PARAMETER #####################

## network architecture
TRANSFORMER 			= True				# use a transformer architecture
N_ENC_LAYERS 			= 4 				# use N layers in the encoder
N_DEC_LAYERS 			= 4 				# use N layers in the decoder

## parameters sharing
SHARE_ENC 				= 3 				# share M=3 out of the N encoder layers
SHARE_DEC 				= 3 				# share M=3 out of the N decoder layers
SHARE_LANG_EMB 			= True 				# share lookup tables
SHARE_OUTPUT_EMB 		= True 				# share projection output layers

## denoising auto-encoder parameters
MONO_DIRECTIONS			= ','.join(LANGS)	# train the auto-encoder on English and French
WORD_SHUFFLE			= 3					# shuffle words
WORD_DROPOUT			= 0.1				# randomly remove words
WORD_BLANK				= 0.2				# randomly blank out words

## back-translation directions (e.g., en->fr->en and fr->en->fr)
PIVO_DIRECTIONS 		= '{0}-{1}-{0},{1}-{0}-{1}'.format(LANGS[0], LANGS[1])

## pretrained embeddings
PRETRAINED_OUT			= True 						# also pretrain output layers

## dynamic loss coefficients
LAMBDA_XE_MONO 			= '0:1,100000:0.1,300000:0'	# auto-encoder loss coefficient
LAMBDA_XE_OTFD 			= 1							# back-translation loss coefficient

## CPU on-the-fly generation
OTF_NUM_PROCESSES 		= 30						# number of CPU jobs for back-parallel data generation
OTF_SYNC_PARAMS_EVERY 	= 1000						# CPU parameters synchronization frequency

## optimization
ENC_OPTIMIZER 			= 'adam,lr=0.0001'			# model optimizer
GROUP_BY_SIZE 			= True 						# group sentences by length inside batches
BATCH_SIZE 				= 32						# batch size
EPOCH_SIZE 				= 500000					# epoch size
# stopping criterion
STOPPING_CRITERION 		= 'bleu_{0}_{1}_valid,10'.format(LANGS[0], LANGS[1])

FREEZE_ENC_EMB 			= False						# freeze encoder embeddings
FREEZE_DEC_EMB 			= False						# freeze decoder embeddings

########################## END TRAINING ##########################

"""
Logger configuration
"""
LOGGER = logging.getLogger(__name__)

def configure_logger(logger):
	logger.setLevel(logging.DEBUG)

	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)
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
PWD = os.path.dirname(os.path.realpath(__file__))

configure_logger(LOGGER)

if len(LANGS) != 2:
	LOGGER.error("exactly two languages are needed")

# Create a abstract dag
LOGGER.info("Creating ADAG: {0}".format(PWD))
dag = ADAG(DAG_ID)

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

	## Transformations
	wrapper_tokenizer = Executable(name="tokenize")
	wrapper_tokenizer.addPFN(PFN("file://"+PWD+"/bin/tokenizer.sh", site="local"))
	
	script_tokenizer = Executable(name="tokenizer")
	script_tokenizer.addPFN(PFN("file://"+PWD+"/bin/tokenizer/tokenizer.perl", site="local"))
	
	normalize_punctuation = Executable(name="normalize-punctuation")
	normalize_punctuation.addPFN(PFN("file://"+PWD+"/bin/tokenizer/normalize-punctuation.perl", site="local"))
	
	x_tokenizer = Transformation(wrapper_tokenizer)
	x_tokenizer.uses(script_tokenizer)
	x_tokenizer.uses(normalize_punctuation)
	
	## End transformations

	tokenize.append(Job(wrapper_tokenizer))
	lang_tok.append(File("{0}.tok".format(lang_raw.name)))
	tokenize[lang].addArguments("-i", lang_raw.name, "-l", LANGS[lang], "-p", str(N_THREADS), "-o", lang_tok[lang].name)


	tokenize[lang].uses(lang_raw, link=Link.INPUT)
	tokenize[lang].uses(lang_tok[lang], link=Link.OUTPUT, transfer=True, register=True)
	
	dag.addJob(tokenize[lang])
	dag.addDependency(Dependency(parent=concat[lang], child=tokenize[lang]))

	LOGGER.info("{0} monolingual data tokenized in: {1}".format(LANGS[lang], lang_tok[lang].name))

## learn BPE codes
fast_bpe = Job("learnbpe")
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
	## Apply BPE codes
	apply_bpe.append(Job("applybpe"))
	apply_bpe[lang].addArguments("applybpe", str(CODES), " ".join([x.name for x in src_tok]))
	
	tok_codes.append(File("{0}.{1}".format(lang_tok[lang].name, str(CODES))))
	apply_bpe[lang].uses(bpe_codes, link=Link.INPUT)
	apply_bpe[lang].uses(lang_tok[lang], link=Link.INPUT)
	apply_bpe[lang].uses(tok_codes[lang], link=Link.OUTPUT, transfer=True, register=True)
	
	dag.addJob(apply_bpe[lang])
	dag.addDependency(Dependency(parent=fast_bpe, child=apply_bpe[lang]))

	LOGGER.info("BPE codes applied to {0} in: {1}".format(LANGS[lang], tok_codes[lang].name))

	## Extract vocabulary for each language
	extract_vocab.append(Job("getvocab"))
	extract_vocab[lang].addArguments("getvocab", tok_codes[lang].name)
	lang_vocab.append(File("vocab.{0}.{1}".format(LANGS[lang], str(CODES))))

	extract_vocab[lang].uses(tok_codes[lang], link=Link.INPUT)
	extract_vocab[lang].uses(lang_vocab[lang], link=Link.OUTPUT, transfer=True, register=True)
	extract_vocab[lang].setStdout(lang_vocab[lang])

	dag.addJob(extract_vocab[lang])
	dag.addDependency(Dependency(parent=apply_bpe[lang], child=extract_vocab[lang]))

	LOGGER.info("{0} vocab in: {1}".format(LANGS[lang], lang_vocab[lang].name))


## Extract vocabulary for all languages
extract_vocab_all = Job("getvocab")
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


###########################################################################
################### parallel data (for evaluation only) ###################
###########################################################################

## TODO

# echo "Extracting parallel data..."
# tar -xzf dev.tgz

# echo "Tokenizing valid and test data..."
# $INPUT_FROM_SGM < $SRC_VALID.sgm | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_VALID
# $INPUT_FROM_SGM < $TGT_VALID.sgm | $NORM_PUNC -l fr | $REM_NON_PRINT_CHAR | $TOKENIZER -l fr -no-escape -threads $N_THREADS > $TGT_VALID
# $INPUT_FROM_SGM < $SRC_TEST.sgm | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_TEST
# $INPUT_FROM_SGM < $TGT_TEST.sgm | $NORM_PUNC -l fr | $REM_NON_PRINT_CHAR | $TOKENIZER -l fr -no-escape -threads $N_THREADS > $TGT_TEST

# echo "Applying BPE to valid and test files..."
# $FASTBPE applybpe $SRC_VALID.$CODES $SRC_VALID $BPE_CODES $SRC_VOCAB
# $FASTBPE applybpe $TGT_VALID.$CODES $TGT_VALID $BPE_CODES $TGT_VOCAB
# $FASTBPE applybpe $SRC_TEST.$CODES $SRC_TEST $BPE_CODES $SRC_VOCAB
# $FASTBPE applybpe $TGT_TEST.$CODES $TGT_TEST $BPE_CODES $TGT_VOCAB

# echo "Binarizing data..."
# rm -f $SRC_VALID.$CODES.pth $TGT_VALID.$CODES.pth $SRC_TEST.$CODES.pth $TGT_TEST.$CODES.pth
# $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_VALID.$CODES
# $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_VALID.$CODES
# $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TEST.$CODES
# $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TEST.$CODES



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

# LOGGER.info("Pre-training fastText on: {0}".format(lang_bpe_all.name))

# fasttext = Job("fasttext")
# bpe_vec = File("{0}.vec".format(lang_bpe_all.name))
# fasttext.addArguments(
# 	"skipgram", "-epoch", str(N_EPOCHS), 
# 	"-minCount", "0", "-dim", "512", "-thread", 
# 	str(N_THREADS), "-ws", "5", "-neg", "10", 
# 	"-input", lang_bpe_all.name, 
# 	"-output", lang_bpe_all.name
# )

# fasttext.uses(lang_bpe_all, link=Link.INPUT)
# fasttext.uses(bpe_vec, link=Link.OUTPUT, transfer=True, register=True)
# dag.addJob(fasttext)
# dag.addDependency(Dependency(parent=concat_bpe, child=fasttext))

# LOGGER.info("Cross-lingual embeddings in: {0}".format(bpe_vec.name))


# ###########################################################################
# ################################ Training #################################
# ###########################################################################

# MONO_DATASET = "'{0}:{1},,;{2}:{3},,'".format(LANGS[0], lang_binarized[0], LANGS[1], lang_binarized[1]) 
# # PARA_DATASET = "'en-fr:,./data/para/dev/newstest2013-ref.XX.60000.pth,./data/para/dev/newstest2014-fren-src.XX.60000.pth'", 

# training = Job("training")
# training_out = File("trained-{0}-{1}.out".format(LANGS[0], LANGS[1]))

# try:
# 	training.addArguments(
# 		'--exp_name', str(DAG_ID), 
# 		'--transformer', str(TRANSFORMER), 
# 		'--n_enc_layers', str(N_ENC_LAYERS), 
# 		'--n_dec_layers', str(N_DEC_LAYERS), 
# 		'--share_enc', str(SHARE_ENC), 
# 		'--share_dec', str(SHARE_DEC), 
# 		'--share_lang_emb', str(SHARE_LANG_EMB), 
# 		'--share_output_emb', str(SHARE_OUTPUT_EMB), 
# 		'--langs', str(MONO_DIRECTIONS), 
# 		'--n_mono', '-1', 
# 		'--mono_directions', str(MONO_DIRECTIONS),
# 		'--mono_dataset', str(MONO_DATASET), 
# 		# '--para_dataset', "'en-fr:,./data/para/dev/newstest2013-ref.XX.60000.pth,./data/para/dev/newstest2014-fren-src.XX.60000.pth'", 
# 		'--word_shuffle', str(WORD_SHUFFLE), 
# 		'--word_dropout', str(WORD_DROPOUT), 
# 		'--word_blank', str(WORD_BLANK), 
# 		'--pivo_directions', str(PIVO_DIRECTIONS), 
# 		'--pretrained_emb', bpe_vec.name, 
# 		'--pretrained_out', str(PRETRAINED_OUT), 
# 		'--lambda_xe_mono', str(LAMBDA_XE_MONO), 
# 		'--lambda_xe_otfd', str(LAMBDA_XE_OTFD), 
# 		'--otf_num_processes', str(OTF_NUM_PROCESSES), 
# 		'--otf_sync_params_every', str(OTF_SYNC_PARAMS_EVERY), 
# 		'--enc_optimizer', str(ENC_OPTIMIZER), 
# 		'--epoch_size', str(EPOCH_SIZE), 
# 		'--stopping_criterion', str(STOPPING_CRITERION)
# 	)

# except FormatError as e:
# 	LOGGER.error("Invalid argument given to the trainer:")
# 	error_lines = traceback.format_exc().splitlines()
# 	for err in error_lines:
# 		if "--" in err:
# 			tmp=STOPPING_CRITERION
# 			LOGGER.error("\t {0}".format(err))

# 	exit(-1)

# dag.addJob(training)

# training.uses(bpe_vec.name, link=Link.INPUT)
# dag.addDependency(Dependency(parent=fasttext, child=training))

# for lang in range(len(LANGS)):
# 	training.uses(lang_binarized[lang], link=Link.INPUT)
# 	dag.addDependency(Dependency(parent=binarize[lang], child=training))


# training.uses(training_out, link=Link.OUTPUT, transfer=True, register=True)
# training.setStdout(training_out)

# LOGGER.info("Model trained => {0}".format(training_out.name))

###########################################################################
#############################  End Training ###############################
###########################################################################


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

# Write the DAX to stdout
LOGGER.info("Writing {}".format(daxfile))

with open(daxfile, "w") as f:
	dag.writeXML(f)

LOGGER.info("Done")

