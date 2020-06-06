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

USER = pwd.getpwuid(os.getuid())[0]
PWD = os.path.dirname(os.path.realpath(__file__))

######################## WORKFLOW PARAMETER ########################

DAG_ID					= "fb-nlp-nmt"
DATA_PATH				= "input/"
CONTAINER				= "fb_nlp"

########################### END WORKFLOW ###########################

######################## PREPROCESS PARAMETER ######################

MONO_PATH				= PWD + '/' + DATA_PATH + "mono"
PARA_PATH				= PWD + '/' + DATA_PATH + "para"
TEST_DATA				= "dev.tgz"

### CAUTION: LANGS      = [src, target]
LANGS					= ['en', 'fr']
YEARS					= [2007, 2008]

# If we need to fetch data from server
BASE_URL				= "http://www.statmt.org/wmt14/training-monolingual-news-crawl/"

########################## END PREPROCESS #########################

####################### PRETRAINING PARAMETER #####################

N_MONO					= 10000000			# number of monolingual sentences for each language
CODES 					= 60000				# number of BPE codes
N_THREADS 				= 16				# number of threads in data preprocessing
N_EPOCHS				= 1					# number of fastText epochs

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

configure_logger(LOGGER)

if len(LANGS) != 2:
	LOGGER.error("exactly two languages are needed")

# Create a abstract dag
LOGGER.info("Creating ADAG: {0}".format(PWD))
dag = ADAG(DAG_ID)

# Add some workflow-level metadata
dag.metadata("creator", "%s@%s" % (USER, os.uname()[1]))
dag.metadata("created", time.ctime())


######################## EXECUTABLES ########################

exe_wget = Executable("wget", installed=True)
exe_wget.addPFN(PFN("/bin/wget", site="local"))
exe_wget.addPFN(PFN("/bin/wget", site="condorpool"))
dag.addExecutable(exe_wget)

exe_gzip = Executable("gzip", installed=True)
exe_gzip.addPFN(PFN("/bin/gunzip", site="local"))
exe_gzip.addPFN(PFN("/bin/gunzip", site="condorpool"))
exe_gzip.addProfile(Profile(Namespace.CONDOR, "request_memory", "10G")); # in MB
dag.addExecutable(exe_gzip)

exe_concat = Executable("concat", installed=False)
exe_concat.addPFN(PFN("file://"+PWD+"/bin/concatenate.sh", site="local"))
exe_concat.addPFN(PFN("file://"+PWD+"/bin/concatenate.sh", site="condorpool"))
exe_concat.addProfile(Profile(Namespace.CONDOR, "request_memory", "10G")); # in MB
dag.addExecutable(exe_concat)

exe_concat_bpe = Executable("concat-bpe", installed=False)
exe_concat_bpe.addPFN(PFN("file://"+PWD+"/bin/concat-bpe.sh", site="local"))
exe_concat_bpe.addPFN(PFN("file://"+PWD+"/bin/concat-bpe.sh", site="condorpool"))
exe_concat_bpe.addProfile(Profile(Namespace.CONDOR, "request_memory", "10G")); # in MB
dag.addExecutable(exe_concat_bpe)

wget = {}
unzip = {}
dataset = {}
concat = {}
tokenize = {}
src_tok = {}
lang_tok = {}

files_already_there = [os.path.basename(x) for x in glob.glob(MONO_PATH+"/*")]

for lang in LANGS:
	wget[lang] = {}
	unzip[lang] = {}
	dataset[lang] = {}

	for year in YEARS:
		current_input = File("news.{1}.{2}.shuffled.gz".format(BASE_URL, year, lang))

		#If the data set is already there
		if current_input.name not in files_already_there:
			LOGGER.info("{} will be downloaded from {}..".format(current_input.name, BASE_URL))
			wget[lang][year] = Job("wget")
			wget[lang][year].addArguments("-c", "{0}/{1}".format(BASE_URL, current_input.name))
			wget[lang][year].uses(current_input, link=Link.OUTPUT, transfer=False, register=False)
			dag.addJob(wget[lang][year])
		else:
			LOGGER.info("{0} found in {1}".format(current_input.name, MONO_PATH))

		dataset[lang][year] = File(current_input.name[:-3])

		# if the data set is already unzipped
		if dataset[lang][year].name not in files_already_there:
			# gunzip
			unzip[lang][year] = Job("gzip")

			unzip[lang][year].uses(current_input, link=Link.INPUT)
			unzip[lang][year].uses(dataset[lang][year], link=Link.OUTPUT, transfer=False, register=False)

			dag.addJob(unzip[lang][year])
			unzip[lang][year].addArguments(current_input.name)

			# Add dependency only of we download the datasets
			if current_input.name not in files_already_there:
				dag.addDependency(Dependency(parent=wget[lang][year], child=unzip[lang][year]))
		else:
			LOGGER.info("{0} already unzipped in {1}".format(current_input.name, MONO_PATH))

	## Concatenate data for each language

	concat[lang] = Job("concat")
	lang_raw = File("all.{0}".format(lang))
	concat[lang].addArguments("-m", str(N_MONO), "-o", lang_raw.name, " ".join([v.name for u,v in dataset[lang].items()]))

	concat[lang].uses(lang_raw, link=Link.OUTPUT, transfer=True, register=True)
	
	dag.addJob(concat[lang])

	for year in YEARS:
		concat[lang].uses(dataset[lang][year], link=Link.INPUT)
		dag.addDependency(Dependency(parent=unzip[lang][year], child=concat[lang]))

	LOGGER.info("{0} monolingual data concatenated in: {1}".format(lang, lang_raw.name))

	## Tokenize each language
	tokenize[lang] = Job("tokenize")
	lang_tok[lang] = File("{0}.tok".format(lang_raw.name))
	tokenize[lang].addArguments("-i", lang_raw.name, "-l", lang, "-p", str(N_THREADS), "-o", lang_tok[lang].name)

	tokenize[lang].uses(lang_raw, link=Link.INPUT)
	tokenize[lang].uses(lang_tok[lang], link=Link.OUTPUT, transfer=True, register=True)
	
	dag.addJob(tokenize[lang])
	dag.addDependency(Dependency(parent=concat[lang], child=tokenize[lang]))

	LOGGER.info("{0} monolingual data tokenized in: {1}".format(lang, lang_tok[lang].name))

## learn BPE codes
fast_bpe = Job("learnbpe")
fast_bpe.addArguments("learnbpe", str(CODES), " ".join([v.name for u,v in lang_tok.items()]))
bpe_codes = File("bpe_codes")
fast_bpe.setStdout(bpe_codes)
dag.addJob(fast_bpe)

for lang in LANGS:
	fast_bpe.uses(lang_tok[lang], link=Link.INPUT)
	dag.addDependency(Dependency(parent=tokenize[lang], child=fast_bpe))

fast_bpe.uses(bpe_codes, link=Link.OUTPUT, transfer=True, register=True)

LOGGER.info("Learning BPE codes")

apply_bpe = {}
tok_codes = {}
extract_vocab = {}
lang_vocab = {}

for lang in LANGS:
	## Apply BPE codes
	apply_bpe[lang] = Job("applybpe")
	
	tok_codes[lang] = File("{0}.{1}".format(lang_tok[lang].name, str(CODES)))
	apply_bpe[lang].uses(bpe_codes, link=Link.INPUT)
	apply_bpe[lang].uses(lang_tok[lang], link=Link.INPUT)
	apply_bpe[lang].uses(tok_codes[lang], link=Link.OUTPUT, transfer=True, register=True)
	
	apply_bpe[lang].addArguments("applybpe", tok_codes[lang].name, lang_tok[lang].name, bpe_codes.name)

	dag.addJob(apply_bpe[lang])
	dag.addDependency(Dependency(parent=fast_bpe, child=apply_bpe[lang]))

	LOGGER.info("BPE codes applied to {0} in: {1}".format(lang, tok_codes[lang].name))

	## Extract vocabulary for each language
	extract_vocab[lang] = Job("getvocab")
	lang_vocab[lang] = File("vocab.{0}.{1}".format(lang, str(CODES)))

	extract_vocab[lang].addArguments("getvocab", tok_codes[lang].name)
	extract_vocab[lang].uses(tok_codes[lang], link=Link.INPUT)
	extract_vocab[lang].uses(lang_vocab[lang], link=Link.OUTPUT, transfer=True, register=True)
	extract_vocab[lang].setStdout(lang_vocab[lang])

	dag.addJob(extract_vocab[lang])
	dag.addDependency(Dependency(parent=apply_bpe[lang], child=extract_vocab[lang]))

	LOGGER.info("{0} vocab in: {1}".format(lang, lang_vocab[lang].name))

## Extract vocabulary for all languages
extract_vocab_all = Job("getvocab")
extract_vocab_all.addArguments("getvocab", " ".join([v.name for u,v in tok_codes.items()]))
lang_vocab_all = File("vocab.{0}.{1}".format("-".join([x for x in LANGS]), str(CODES)))
dag.addJob(extract_vocab_all)
extract_vocab_all.setStdout(lang_vocab_all)

for lang in LANGS:
	extract_vocab_all.uses(tok_codes[lang], link=Link.INPUT)
	dag.addDependency(Dependency(parent=apply_bpe[lang], child=extract_vocab_all))

extract_vocab_all.uses(lang_vocab_all, link=Link.OUTPUT, transfer=True, register=True)

LOGGER.info("Full vocab in: {0}".format(lang_vocab_all.name))


## Binarize data
binarize = {}
lang_binarized = {}

for lang in LANGS:
	binarize[lang] = Job("binarize")
	binarize[lang].addArguments(lang_vocab_all.name, tok_codes[lang].name)

	lang_binarized[lang] = File("{0}.pth".format(tok_codes[lang].name))

	binarize[lang].uses(lang_vocab_all, link=Link.INPUT)
	binarize[lang].uses(tok_codes[lang], link=Link.INPUT)

	binarize[lang].uses(lang_binarized[lang], link=Link.OUTPUT, transfer=True, register=True)
	#binarize[lang].setStdout(lang_binarized[lang])

	dag.addJob(binarize[lang])
	dag.addDependency(Dependency(parent=extract_vocab_all, child=binarize[lang]))
	dag.addDependency(Dependency(parent=apply_bpe[lang], child=binarize[lang]))

	LOGGER.info("{0} binarized data in: {1}".format(lang, lang_binarized[lang].name))

###########################################################################
################### parallel data (for evaluation only) ###################
###########################################################################

input_dev = File(TEST_DATA)
# data_dev = File(input_dev.name.split('.')[0]) #Remove extension

# unzip_dev = Job("gzip")

# unzip_dev.uses(input_dev, link=Link.INPUT)
# unzip_dev.uses(data_dev, link=Link.OUTPUT, transfer=False, register=False)

# dag.addJob(unzip_dev)
# unzip_dev.addArguments(input_dev)
# LOGGER.info("Parallel test data {0} unzipped in: {1}".format(input_dev.name, data_dev.name))

## Tokenizing valid and test data
job_valid = {}
file_valid = {}
file_valid_sgm = {}

job_test = {}
file_test = {}
file_test_sgm = {}

for lang in LANGS:
	job_valid[lang] = Job("tokenize-validation")

	file_valid[lang] = File('newstest2013-ref.{0}'.format(lang))
	job_valid[lang].uses(input_dev, link=Link.INPUT)
	job_valid[lang].uses(file_valid[lang], link=Link.OUTPUT, transfer=True, register=True)
	file_valid_sgm[lang] = File('{0}.sgm'.format(file_valid[lang].name))

	dag.addJob(job_valid[lang])
	job_valid[lang].addArguments("-i", file_valid_sgm[lang].name, "-l", lang, "-p", str(N_THREADS), "-o", file_valid[lang].name)

	# dag.addDependency(Dependency(parent=unzip_dev, child=job_valid[lang]))
	LOGGER.info("Tokenizing valid {0} data {1}".format(lang, file_valid[lang].name))

	# Tokenizing test source data
	job_test[lang] = Job("tokenize-validation")
	job_test[lang].uses(input_dev, link=Link.INPUT)

	file_test[lang] = File('newstest2014-{0}-src.{1}'.format(''.join(reversed(LANGS)),lang))
	job_test[lang].uses(file_test[lang], link=Link.OUTPUT, transfer=True, register=True)
	file_test_sgm[lang] = File('{0}.sgm'.format(file_test[lang].name))
	job_test[lang].addArguments("-i", file_test_sgm[lang].name, "-l", lang, "-p", str(N_THREADS), "-o", file_test[lang].name)

	dag.addJob(job_test[lang])

	# dag.addDependency(Dependency(parent=unzip_dev, child=job_test[lang]))
	LOGGER.info("Tokenizing test {0} data {1}".format(lang, file_test[lang].name))


## Applying BPE to valid and test files

# echo "Applying BPE to valid and test files..."
# $FASTBPE applybpe $SRC_VALID.$CODES $SRC_VALID $BPE_CODES $SRC_VOCAB
# $FASTBPE applybpe $TGT_VALID.$CODES $TGT_VALID $BPE_CODES $TGT_VOCAB
# $FASTBPE applybpe $SRC_TEST.$CODES $SRC_TEST $BPE_CODES $SRC_VOCAB
# $FASTBPE applybpe $TGT_TEST.$CODES $TGT_TEST $BPE_CODES $TGT_VOCAB

job_apply_valid = {}
file_apply_valid = {}
job_apply_test = {}
file_apply_test = {}

for lang in LANGS:
	## Apply BPE codes for validation
	job_apply_valid[lang] = Job("applybpe")
	
	file_apply_valid[lang] = File("{0}.{1}".format(file_valid[lang].name, str(CODES)))
	job_apply_valid[lang].uses(bpe_codes, link=Link.INPUT)
	job_apply_valid[lang].uses(file_valid[lang], link=Link.INPUT)
	job_apply_valid[lang].uses(lang_vocab[lang], link=Link.INPUT)
	job_apply_valid[lang].uses(file_apply_valid[lang], link=Link.OUTPUT, transfer=True, register=True)
	
	job_apply_valid[lang].addArguments("applybpe", file_apply_valid[lang].name, file_valid[lang].name, bpe_codes.name, lang_vocab[lang].name)

	dag.addJob(job_apply_valid[lang])
	dag.addDependency(Dependency(parent=fast_bpe, child=job_apply_valid[lang]))
	dag.addDependency(Dependency(parent=job_valid[lang], child=job_apply_valid[lang]))
	dag.addDependency(Dependency(parent=extract_vocab[lang], child=job_apply_valid[lang]))

	LOGGER.info("BPE codes for validation applied to {0} in: {1}".format(lang, file_apply_valid[lang].name))

	## Apply BPE codes for test
	job_apply_test[lang] = Job("applybpe")
	
	file_apply_test[lang] = File("{0}.{1}".format(file_test[lang].name, str(CODES)))
	job_apply_test[lang].uses(bpe_codes, link=Link.INPUT)
	job_apply_test[lang].uses(file_test[lang], link=Link.INPUT)
	job_apply_test[lang].uses(lang_vocab[lang], link=Link.INPUT)
	job_apply_test[lang].uses(file_apply_test[lang], link=Link.OUTPUT, transfer=True, register=True)
	
	job_apply_test[lang].addArguments("applybpe", file_apply_test[lang].name, file_test[lang].name, bpe_codes.name, lang_vocab[lang].name)

	dag.addJob(job_apply_test[lang])
	dag.addDependency(Dependency(parent=fast_bpe, child=job_apply_test[lang]))
	dag.addDependency(Dependency(parent=job_test[lang], child=job_apply_test[lang]))
	dag.addDependency(Dependency(parent=extract_vocab[lang], child=job_apply_test[lang]))

	LOGGER.info("BPE codes for test applied to {0} in: {1}".format(lang, file_apply_test[lang].name))


## Binarizing data
job_binarize_valid = {}
file_binarize_valid = {}
job_binarize_test = {}
file_binarize_test = {}

for lang in LANGS:
	## Binarize for valid data
	job_binarize_valid[lang] = Job("binarize")
	job_binarize_valid[lang].addArguments(lang_vocab_all.name, file_apply_valid[lang].name)

	file_binarize_valid[lang] = File("{0}.pth".format(file_apply_valid[lang].name))

	job_binarize_valid[lang].uses(lang_vocab_all, link=Link.INPUT)
	job_binarize_valid[lang].uses(file_apply_valid[lang], link=Link.INPUT)
	job_binarize_valid[lang].uses(file_binarize_valid[lang], link=Link.OUTPUT, transfer=True, register=True)

	dag.addJob(job_binarize_valid[lang])
	dag.addDependency(Dependency(parent=extract_vocab_all, child=job_binarize_valid[lang]))
	dag.addDependency(Dependency(parent=job_apply_valid[lang], child=job_binarize_valid[lang]))

	LOGGER.info("{0} binarized valid data in: {1}".format(lang, file_binarize_valid[lang].name))

	## Binarize for test data
	job_binarize_test[lang] = Job("binarize")
	job_binarize_test[lang].addArguments(lang_vocab_all.name, file_apply_test[lang].name)

	file_binarize_test[lang] = File("{0}.pth".format(file_apply_test[lang].name))

	job_binarize_test[lang].uses(lang_vocab_all, link=Link.INPUT)
	job_binarize_test[lang].uses(file_apply_test[lang], link=Link.INPUT)
	job_binarize_test[lang].uses(file_binarize_test[lang], link=Link.OUTPUT, transfer=True, register=True)

	dag.addJob(job_binarize_test[lang])
	dag.addDependency(Dependency(parent=extract_vocab_all, child=job_binarize_test[lang]))
	dag.addDependency(Dependency(parent=job_apply_test[lang], child=job_binarize_test[lang]))

	LOGGER.info("{0} binarized test data in: {1}".format(lang, file_binarize_test[lang].name))

# Need to replace en and fr by XX for some reasons
# the training script takes a XX and probably replaces it by en and fr internally
corrected_valid = file_binarize_valid[LANGS[0]].name.replace('.'+LANGS[0]+'.', ".XX.")
corrected_test = file_binarize_test[LANGS[0]].name.replace('.'+LANGS[0]+'.', ".XX.")
LOGGER.info("Parallel data set files:")
LOGGER.info("\t\t validation => {0}".format(corrected_valid))
LOGGER.info("\t\t test       => {0}".format(corrected_test))

###########################################################################
################# Pre-training on concatenated embeddings #################
###########################################################################

## Concatenating source and target monolingual data
concat_bpe = Job("concat-bpe")
lang_bpe_all = File("all.{0}.{1}".format("-".join([x for x in LANGS]), str(CODES)))
concat_bpe.addArguments("-o", lang_bpe_all.name, " ".join([v.name for u,v in tok_codes.items()]))

concat_bpe.uses(lang_bpe_all, link=Link.OUTPUT, transfer=True, register=True)
dag.addJob(concat_bpe)

for lang in LANGS:
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

MONO_DATASET = "'{0}:{1},,;{2}:{3},,'".format(LANGS[0], lang_binarized[LANGS[0]].name, LANGS[1], lang_binarized[LANGS[1]].name) 
# PARA_DATASET = "'{0}:,./data/para/dev/newstest2013-ref.XX.60000.pth,./data/para/dev/newstest2014-fren-src.XX.60000.pth'"
PARA_DATASET = "'{0}:,{1},{2}'".format('-'.join(LANGS), corrected_valid, corrected_test)
# Pretrained model produced by fasttext task
PRETRAINED = bpe_vec.name

training = Job("training")
training_out = File("trained-{0}-{1}.out".format(LANGS[0], LANGS[1]))

try:
	training.addArguments(
		'--exp_name', str(DAG_ID), 
		'--transformer', str(TRANSFORMER), 
		'--n_enc_layers', str(N_ENC_LAYERS), 
		'--n_dec_layers', str(N_DEC_LAYERS), 
		'--share_enc', str(SHARE_ENC), 
		'--share_dec', str(SHARE_DEC), 
		'--share_lang_emb', str(SHARE_LANG_EMB), 
		'--share_output_emb', str(SHARE_OUTPUT_EMB), 
		'--langs', str(MONO_DIRECTIONS), 
		'--n_mono', '-1',
		'--mono_directions', str(MONO_DIRECTIONS),
		'--mono_dataset', str(MONO_DATASET), 
		'--para_dataset', str(PARA_DATASET), 
		'--word_shuffle', str(WORD_SHUFFLE), 
		'--word_dropout', str(WORD_DROPOUT), 
		'--word_blank', str(WORD_BLANK), 
		'--pivo_directions', str(PIVO_DIRECTIONS), 
		'--pretrained_emb', str(PRETRAINED), 
		'--pretrained_out', str(PRETRAINED_OUT), 
		'--lambda_xe_mono', str(LAMBDA_XE_MONO), 
		'--lambda_xe_otfd', str(LAMBDA_XE_OTFD), 
		'--otf_num_processes', str(OTF_NUM_PROCESSES), 
		'--otf_sync_params_every', str(OTF_SYNC_PARAMS_EVERY), 
		'--enc_optimizer', str(ENC_OPTIMIZER), 
		'--epoch_size', str(EPOCH_SIZE), 
		'--stopping_criterion', str(STOPPING_CRITERION)
	)

except FormatError as e:
	LOGGER.error("Invalid argument given to the trainer:")
	error_lines = traceback.format_exc().splitlines()
	for err in error_lines:
		if "--" in err:
			tmp=STOPPING_CRITERION
			LOGGER.error("\t {0}".format(err))

	exit(-1)

dag.addJob(training)

training.uses(bpe_vec.name, link=Link.INPUT)
dag.addDependency(Dependency(parent=fasttext, child=training))

for lang in LANGS:
	training.uses(lang_binarized[lang], link=Link.INPUT)
	training.uses(file_binarize_valid[lang], link=Link.INPUT)
	training.uses(file_binarize_test[lang], link=Link.INPUT)

	dag.addDependency(Dependency(parent=binarize[lang], child=training))	
	dag.addDependency(Dependency(parent=job_binarize_valid[lang], child=training))
	dag.addDependency(Dependency(parent=job_binarize_test[lang], child=training))


training.uses(training_out, link=Link.OUTPUT, transfer=True, register=True)
training.setStdout(training_out)

LOGGER.info("Model trained => {0}".format(training_out.name))

###########################################################################
#############################  End Training ###############################
###########################################################################

# Write the DAX to stdout
LOGGER.info("Writing {}".format(daxfile))

with open(daxfile, "w") as f:
	dag.writeXML(f)

LOGGER.info("Done")

