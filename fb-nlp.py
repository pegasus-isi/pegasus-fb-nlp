#!/usr/bin/env python3
import os
import logging
import sys
import subprocess
import glob
import argparse
import traceback

from logging import Logger
from datetime import datetime
from pathlib import Path

# --- Import Pegasus API -------------------------------------------------------
from Pegasus.api import *

# --- Work Directory Setup -----------------------------------------------------
RUN_ID                  = "fb-nlp-" + datetime.now().strftime("%s")
TOP_DIR                 = Path.cwd()
WORK_DIR                = TOP_DIR / "work"

######################## WORKFLOW PARAMETER ########################

DATA_PATH               = "input/"
CONTAINER               = "fb-nlp"

########################### END WORKFLOW ###########################

######################## PREPROCESS PARAMETER ######################

MONO_PATH               = TOP_DIR / DATA_PATH / "mono"
PARA_PATH               = TOP_DIR / DATA_PATH / "para"
TEST_DATA               = "dev.tgz"

### CAUTION ORDER: LANGS      = [src, target]
LANGS                   = ['en', 'fr']
YEARS                   = [2007, 2008]

# If we need to fetch data from server
BASE_URL                = "http://www.statmt.org/wmt14/training-monolingual-news-crawl/"

########################## END PREPROCESS #########################

####################### PRETRAINING PARAMETER #####################

N_MONO                  = 10000000                  # number of monolingual sentences for each language
CODES                   = 60000                     # number of BPE codes
N_EPOCHS                = 1                         # number of fastText epochs

########################## END PRETRAINING #######################

######################### TRAINING PARAMETER #####################

## network architecture
TRANSFORMER             = True                      # use a transformer architecture
N_ENC_LAYERS            = 4                         # use N layers in the encoder
N_DEC_LAYERS            = 4                         # use N layers in the decoder

## parameters sharing
SHARE_ENC               = 3                         # share M=3 out of the N encoder layers
SHARE_DEC               = 3                         # share M=3 out of the N decoder layers
SHARE_LANG_EMB          = True                      # share lookup tables
SHARE_OUTPUT_EMB        = True                      # share projection output layers

## denoising auto-encoder parameters
MONO_DIRECTIONS         = ','.join(LANGS)           # train the auto-encoder on English and French
WORD_SHUFFLE            = 3                         # shuffle words
WORD_DROPOUT            = 0.1                       # randomly remove words
WORD_BLANK              = 0.2                       # randomly blank out words

## back-translation directions (e.g., en->fr->en and fr->en->fr)
PIVO_DIRECTIONS         = '{0}-{1}-{0},{1}-{0}-{1}'.format(LANGS[0], LANGS[1])

## pretrained embeddings
PRETRAINED_OUT          = True                          # also pretrain output layers

## dynamic loss coefficients
LAMBDA_XE_MONO          = '0:1,100000:0.1,300000:0'     # auto-encoder loss coefficient
LAMBDA_XE_OTFD          = 1                             # back-translation loss coefficient

## CPU on-the-fly generation
OTF_NUM_PROCESSES       = 30                            # number of CPU jobs for back-parallel data generation
OTF_SYNC_PARAMS_EVERY   = 1000                          # CPU parameters synchronization frequency

## optimization
ENC_OPTIMIZER           = 'adam,lr=0.0001'              # model optimizer
GROUP_BY_SIZE           = True                          # group sentences by length inside batches
BATCH_SIZE              = 32                            # batch size
EPOCH_SIZE              = 500000                        # epoch size
# stopping criterion
STOPPING_CRITERION      = 'bleu_{0}_{1}_valid,10'.format(LANGS[0], LANGS[1])

FREEZE_ENC_EMB          = False                        # freeze encoder embeddings
FREEZE_DEC_EMB          = False                        # freeze decoder embeddings

########################## END TRAINING ##########################

"""
Logger configuration
"""
LOGGER = logging.getLogger(__name__)

if len(LANGS) != 2:
    LOGGER.error("exactly two languages must be provided: [source, destination]")

try:
    Path.mkdir(WORK_DIR)
except FileExistsError:
    pass

class WorkflowNLP():
    def __init__(self, name, threads=1, gpus=1, is_singularity=False, training=False, logger=None):
        self.properties = Properties()
        self.site_catalog = SiteCatalog()
        self.transformation_catalog = TransformationCatalog()
        self.replica_catalog = ReplicaCatalog()
        self.workflow = Workflow(name, infer_dependencies=True)
        self.logger = logger
        self.threads = threads
        self.gpus = gpus
        self.training = training
        self.is_singularity = is_singularity
        
        self.input_dev = File(TEST_DATA)

        self.logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if self.training:
            self.logger.info("Training ACTIVATED.")
            self.logger.info("This workflow uses {0} threads for: \"tokenize\", \"tokenize-validation\", \"fasttext\" and \"training\" tasks.".format(self.threads))
        else:
            self.logger.info("This workflow uses {0} threads for: \"tokenize\", \"tokenize-validation\" and \"fasttext\" tasks.".format(self.threads))



    def set_properties(self):
        # --- Configuration (Pegasus Properties) ---------------------------------------

        self.properties["pegasus.data.configuration"] = "condorio"
        # props["pegasus.monitord.encoding"] = "json"                                                                    
        self.properties["pegasus.integrity.checking"] = "none"

        # pegasus-planner will, by default, pick up this file in cwd
        self.properties.write()
        self.logger.info("Creating properties")

    def set_sites(self):
        # --- Site Catalog -------------------------------------------------------------
        shared_scratch_dir = str(WORK_DIR / RUN_ID)
        local_storage_dir = str(WORK_DIR / "outputs" / RUN_ID)

        local = Site("local")\
                    .add_directories(
                        Directory(Directory.SHARED_SCRATCH, shared_scratch_dir)
                            .add_file_servers(FileServer("file://" + shared_scratch_dir, Operation.ALL)),
                        
                        Directory(Directory.LOCAL_STORAGE, local_storage_dir)
                            .add_file_servers(FileServer("file://" + local_storage_dir, Operation.ALL))
                    )

        condorpool = Site("condorpool") \
                        .add_pegasus_profile(style="condor") \
                        .add_condor_profile(universe="vanilla")

        self.site_catalog.add_sites(local, condorpool)

        # pegasus-planner will, by default, pick up this file in cwd
        self.site_catalog.write()
        self.logger.info("Creating sites catalog")

    def set_transformations(self):
        # --- Transformation Catalog (Executables and Containers) ----------------------
        # Create and add our container to the TransformationCatalog.

        # A container that will be used to execute the following two transformations.
        if self.is_singularity:
            fb_nlp = Container(
                            CONTAINER, 
                            Container.SINGULARITY,
                            image="library://papajim/default/fb-nlp"
                        )
        else:
            fb_nlp = Container(
                            CONTAINER, 
                            Container.DOCKER, 
                            image="docker:///pegasus/fb-nlp:latest"
                        )

        self.transformation_catalog.add_containers(fb_nlp)

        # Create and add our transformations to the TransformationCatalog.

        exe_wget = Transformation(
                        "wget",
                        site="condorpool",
                        pfn="/usr/bin/wget",
                        is_stageable=False,
                    )

        exe_gzip = Transformation(
                        "gzip",
                        site="condorpool",
                        pfn="/bin/gunzip",
                        is_stageable=False,
                        container=fb_nlp,
                    )

        exe_concat = Transformation(
                        "concat",
                        site="condorpool",
                        pfn=str(Path(__file__).parent.resolve() / "bin/concatenate.sh"),
                        is_stageable=True,
                    )

        exe_concat_bpe = Transformation(
                        "concat-bpe",
                        site="condorpool",
                        pfn=str(Path(__file__).parent.resolve() / "bin/concat-bpe.sh"),
                        is_stageable=True,
                    )

        # All executable that are installed inside of "container".
        exe_tokenize = Transformation(
                        "tokenize",
                        site="condorpool",
                        pfn="file:///tokenize.sh",
                        is_stageable=False,
                        container=fb_nlp
                    )
        exe_tokenize.add_condor_profile(request_cpus=self.threads)
        exe_tokenize.add_condor_profile(request_memory="10 GB")


        exe_tokenize_valid = Transformation(
                        "tokenize-validation",
                        site="condorpool",
                        pfn="file:///tokenize-validation.sh",
                        is_stageable=False,
                        container=fb_nlp
                    )
        exe_tokenize_valid.add_condor_profile(request_cpus=self.threads)
        exe_tokenize_valid.add_condor_profile(request_memory="10 GB")


        exe_bpe = Transformation(
                        "bpe",
                        site="condorpool",
                        pfn="file:///UnsupervisedMT/NMT/tools/fastBPE/fast",
                        is_stageable=False,
                        container=fb_nlp
                    )
        exe_bpe.add_condor_profile(request_memory="10 GB")

        exe_binarize = Transformation(
                        "binarize",
                        site="condorpool",
                        pfn="file:///UnsupervisedMT/NMT/preprocess.py",
                        is_stageable=False,
                        container=fb_nlp
                    )
        exe_binarize.add_condor_profile(request_memory="10 GB")

        exe_fasttext = Transformation(
                        "fasttext",
                        site="condorpool",
                        pfn="file:///UnsupervisedMT/NMT/tools/fastText/fasttext",
                        is_stageable=False,
                        container=fb_nlp
                    )
        exe_fasttext.add_condor_profile(request_cpus=self.threads)
        exe_fasttext.add_condor_profile(request_memory="10 GB")

        exe_training = Transformation(
                        "training",
                        site="condorpool",
                        pfn="file:///UnsupervisedMT/NMT/main.py",
                        is_stageable=False,
                        container=fb_nlp
                    )
        exe_training.add_condor_profile(request_cpus=self.threads)
        exe_training.add_condor_profile(request_memory="10 GB")

        # TODO: stage sh directly into container
        # # A stageable python script that must be executed inside tools_container because
        # # it contains packages that we have when we develop locally, but may not be 
        # # installed on a compute node. 
        # process_text_2nd_pass = Transformation(
        #                             "process_text_2nd_pass.py",
        #                             site="workflow-cloud",
        #                             pfn="http://www.isi.edu/~tanaka/process_text_2nd_pass.py",
        #                             is_stageable=True,
        #                             container=tools_container
        #                         )

        self.transformation_catalog.add_transformations(
                exe_wget,
                exe_gzip,
                exe_concat,
                exe_concat_bpe,
                exe_tokenize,
                exe_tokenize_valid,
                exe_bpe,
                exe_binarize,
                exe_fasttext,
                exe_training
            )

        # pegasus-planner will, by default, pick up this file in cwd
        self.transformation_catalog.write()
        self.logger.info("Creating transformations catalog")

    def set_replicas(self):
        # --- Replica Catalog ----------------------------------------------------------
        # Any initial input files must be specified in the ReplicaCatalog object. In this
        # workflow, we have 1 input file to the workflow, and pegasus needs to know where
        # this file is located. We specify that when calling add_replica().

        self.replica_catalog = ReplicaCatalog() \
                .add_regex_replica("local", ".*\.gz", str(Path(__file__).parent.resolve() / "input/mono/[0]")) \
                .add_replica("local", self.input_dev, str(Path(__file__).parent.resolve() / "input/para/" / self.input_dev.lfn))

        # Again, pegasus-planner will know to look for this file in cwd.
        self.replica_catalog.write()
        self.logger.info("Creating replica catalog")


    def set_jobs(self):
        # --- Workflow -----------------------------------------------------------------
        # Set infer_dependencies=True so that they are inferred based on job
        # input and output file usage.

        wget = {}
        unzip = {}
        dataset = {}
        concat = {}
        tokenize = {}
        src_tok = {}
        lang_tok = {}

        files_already_there = [os.path.basename(x) for x in glob.glob(str(MONO_PATH)+"/*")]

        for lang in LANGS:
            wget[lang] = {}
            unzip[lang] = {}
            dataset[lang] = {}

            for year in YEARS:
                current_input = File("news.{1}.{2}.shuffled.gz".format(BASE_URL, year, lang))

                #If the data set is already there
                if current_input.lfn not in files_already_there:
                    self.logger.info("{} will be downloaded from {}..".format(current_input, BASE_URL))

                    wget[lang][year] = Job("wget") \
                                        .add_outputs(current_input, stage_out=False, register_replica=False) \
                                        .add_args("-c", "{0}/{1}".format(BASE_URL, current_input)) 

                    self.workflow.add_jobs(wget[lang][year])
                else:
                    self.logger.info("{0} found in {1}".format(current_input, MONO_PATH))

                dataset[lang][year] = File(current_input.lfn[:-3])

                # if the data set is already unzipped
                if dataset[lang][year].lfn not in files_already_there:
                    # gunzip
                    unzip[lang][year] = Job("gzip") \
                                            .add_inputs(current_input) \
                                            .add_outputs(dataset[lang][year], stage_out=True, register_replica=True) \
                                            .add_args(current_input)

                    self.workflow.add_jobs(unzip[lang][year])

                    # # Add dependency only of we download the datasets
                    # if current_input.name not in files_already_there:
                    #   dag.addDependency(Dependency(parent=wget[lang][year], child=unzip[lang][year]))
                else:
                    self.logger.info("{0} already unzipped in {1}".format(current_input, MONO_PATH))

            ## Concatenate data for each language

            concat[lang] = Job("concat")
            lang_raw = File("all.{0}".format(lang))
            
            for year in YEARS:
                concat[lang].add_inputs(dataset[lang][year])

            concat[lang].add_outputs(lang_raw)
            concat[lang].add_args("-m", str(N_MONO), "-o", lang_raw, *[v for u,v in dataset[lang].items()])

            self.workflow.add_jobs(concat[lang])
            self.logger.info("{0} monolingual data concatenated in: {1}".format(lang, lang_raw))

            ## Tokenize each language
            tokenize[lang] = Job("tokenize")
            lang_tok[lang] = File("{0}.tok".format(lang_raw))

            tokenize[lang].add_inputs(lang_raw)
            tokenize[lang].add_outputs(lang_tok[lang])
            tokenize[lang].add_args("-i", lang_raw, "-l", lang, "-p", str(self.threads), "-o", lang_tok[lang])

            self.workflow.add_jobs(tokenize[lang])
            self.logger.info("{0} monolingual data tokenized in: {1}".format(lang, lang_tok[lang]))

        ## learn BPE codes
        fast_bpe = Job("bpe")
        bpe_codes = File("bpe_codes")

        for lang in LANGS:
            fast_bpe.add_inputs(lang_tok[lang])

        fast_bpe.set_stdout(bpe_codes)
        fast_bpe.add_args("learnbpe", str(CODES), *[v for u,v in lang_tok.items()])

        self.workflow.add_jobs(fast_bpe)

        self.logger.info("Learning BPE codes")

        apply_bpe = {}
        tok_codes = {}
        extract_vocab = {}
        lang_vocab = {}

        for lang in LANGS:
            ## Apply BPE codes
            apply_bpe[lang] = Job("bpe")
            
            tok_codes[lang] = File("{0}.{1}".format(lang_tok[lang], str(CODES)))

            apply_bpe[lang].add_inputs(bpe_codes, lang_tok[lang])
            apply_bpe[lang].add_outputs(tok_codes[lang])
            apply_bpe[lang].add_args("applybpe", tok_codes[lang], lang_tok[lang], bpe_codes)

            self.workflow.add_jobs(apply_bpe[lang])
            self.logger.info("BPE codes applied to {0} in: {1}".format(lang, tok_codes[lang]))

            ## Extract vocabulary for each language
            extract_vocab[lang] = Job("bpe")
            lang_vocab[lang] = File("vocab.{0}.{1}".format(lang, str(CODES)))

            extract_vocab[lang].add_inputs(tok_codes[lang])
            extract_vocab[lang].set_stdout(lang_vocab[lang])
            extract_vocab[lang].add_args("getvocab", tok_codes[lang])

            self.workflow.add_jobs(extract_vocab[lang])
            self.logger.info("{0} vocab in: {1}".format(lang, lang_vocab[lang]))

        ## Extract vocabulary for all languages
        extract_vocab_all = Job("bpe")
        lang_vocab_all = File("vocab.{0}.{1}".format("-".join([x for x in LANGS]), str(CODES)))

        for lang in LANGS:
            extract_vocab_all.add_inputs(tok_codes[lang])

        extract_vocab_all.set_stdout(lang_vocab_all)
        extract_vocab_all.add_args("getvocab", *[v for u,v in tok_codes.items()])

        self.workflow.add_jobs(extract_vocab_all)
        self.logger.info("Full vocab in: {0}".format(lang_vocab_all))

        ## Binarize data
        binarize = {}
        lang_binarized = {}

        for lang in LANGS:
            binarize[lang] = Job("binarize")
            lang_binarized[lang] = File("{0}.pth".format(tok_codes[lang]))

            binarize[lang].add_inputs(lang_vocab_all, tok_codes[lang])
            binarize[lang].add_outputs(lang_binarized[lang])
            binarize[lang].add_args(lang_vocab_all, tok_codes[lang])

            self.workflow.add_jobs(binarize[lang])

            self.logger.info("{0} binarized data in: {1}".format(lang, lang_binarized[lang]))

            ###########################################################################
            ################### parallel data (for evaluation only) ###################
            ###########################################################################

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
                file_valid_sgm[lang] = File('{0}.sgm'.format(file_valid[lang]))

                job_valid[lang].add_inputs(self.input_dev) # the dev.tgz containing data for validations
                job_valid[lang].add_outputs(file_valid[lang])
                job_valid[lang].add_args("-i", file_valid_sgm[lang], "-l", lang, "-p", str(self.threads), "-o", file_valid[lang])

                self.workflow.add_jobs(job_valid[lang])

                self.logger.info("Tokenizing valid {0} data {1}".format(lang, file_valid[lang]))

                # Tokenizing test source data
                job_test[lang] = Job("tokenize-validation")
                file_test[lang] = File('newstest2014-{0}-src.{1}'.format(''.join(reversed(LANGS)),lang))
                file_test_sgm[lang] = File('{0}.sgm'.format(file_test[lang]))

                job_test[lang].add_inputs(self.input_dev)
                job_test[lang].add_outputs(file_test[lang])
                job_test[lang].add_args("-i", file_test_sgm[lang], "-l", lang, "-p", str(self.threads), "-o", file_test[lang])

                self.workflow.add_jobs(job_test[lang])

                self.logger.info("Tokenizing test {0} data {1}".format(lang, file_test[lang]))

            ## Applying BPE to valid and test files
            job_apply_valid = {}
            file_apply_valid = {}
            job_apply_test = {}
            file_apply_test = {}

            for lang in LANGS:
                ## Apply BPE codes for validation
                job_apply_valid[lang] = Job("bpe")
                file_apply_valid[lang] = File("{0}.{1}".format(file_valid[lang], str(CODES)))

                job_apply_valid[lang].add_inputs(bpe_codes, file_valid[lang], lang_vocab[lang])
                job_apply_valid[lang].add_outputs(file_apply_valid[lang])
                
                job_apply_valid[lang].add_args("applybpe", file_apply_valid[lang], file_valid[lang], bpe_codes, lang_vocab[lang])

                self.workflow.add_jobs(job_apply_valid[lang])
                self.logger.info("BPE codes for validation applied to {0} in: {1}".format(lang, file_apply_valid[lang]))

                ## Apply BPE codes for test
                job_apply_test[lang] = Job("bpe")
                file_apply_test[lang] = File("{0}.{1}".format(file_test[lang], str(CODES)))

                job_apply_test[lang].add_inputs(bpe_codes, file_test[lang], lang_vocab[lang])
                job_apply_test[lang].add_outputs(file_apply_test[lang])
                job_apply_test[lang].add_args("applybpe", file_apply_test[lang], file_test[lang], bpe_codes, lang_vocab[lang])

                self.workflow.add_jobs(job_apply_test[lang])
                self.logger.info("BPE codes for test applied to {0} in: {1}".format(lang, file_apply_test[lang]))

            ## Binarizing data
            job_binarize_valid = {}
            file_binarize_valid = {}
            job_binarize_test = {}
            file_binarize_test = {}

            for lang in LANGS:
                ## Binarize for valid data
                job_binarize_valid[lang] = Job("binarize")
                file_binarize_valid[lang] = File("{0}.pth".format(file_apply_valid[lang]))

                job_binarize_valid[lang].add_inputs(lang_vocab_all, file_apply_valid[lang])
                job_binarize_valid[lang].add_outputs(file_binarize_valid[lang])
                job_binarize_valid[lang].add_args(lang_vocab_all, file_apply_valid[lang])

                self.workflow.add_jobs(job_binarize_valid[lang])
                self.logger.info("{0} binarized valid data in: {1}".format(lang, file_binarize_valid[lang]))

                ## Binarize for test data
                job_binarize_test[lang] = Job("binarize")
                file_binarize_test[lang] = File("{0}.pth".format(file_apply_test[lang]))

                job_binarize_test[lang].add_inputs(lang_vocab_all, file_apply_test[lang])
                job_binarize_test[lang].add_outputs(file_binarize_test[lang])
                job_binarize_test[lang].add_args(lang_vocab_all, file_apply_test[lang])

                self.workflow.add_jobs(job_binarize_test[lang])
                self.logger.info("{0} binarized test data in: {1}".format(lang, file_binarize_test[lang]))

            # the main training task required that we 
            # replace en and fr by XX for some technical reasons
            corrected_valid = file_binarize_valid[LANGS[0]].lfn.replace('.'+LANGS[0]+'.', ".XX.")
            corrected_test = file_binarize_test[LANGS[0]].lfn.replace('.'+LANGS[0]+'.', ".XX.")
            self.logger.info("Parallel data set files:")
            self.logger.info("\t\t validation => {0}".format(corrected_valid))
            self.logger.info("\t\t test       => {0}".format(corrected_test))

        ###########################################################################
        ################# Pre-training on concatenated embeddings #################
        ###########################################################################

        ## Concatenating source and target monolingual data
        concat_bpe = Job("concat-bpe")
        lang_bpe_all = File("all.{0}.{1}".format("-".join([x for x in LANGS]), str(CODES)))

        for lang in LANGS:
            concat_bpe.add_inputs(tok_codes[lang])

        concat_bpe.add_outputs(lang_bpe_all)
        concat_bpe.add_args("-o", lang_bpe_all, *[v for u,v in tok_codes.items()])

        self.workflow.add_jobs(concat_bpe)
        self.logger.info("Concatenated shuffled data in used by the pre-training task fasttext: {0}".format(lang_bpe_all))

        ## Pre-training with fastText

        fasttext = Job("fasttext")
        bpe_vec = File("{0}.vec".format(lang_bpe_all))

        fasttext.add_inputs(lang_bpe_all)
        fasttext.add_outputs(bpe_vec)
        fasttext.add_args(
            "skipgram", 
            "-epoch", str(N_EPOCHS), 
            "-minCount", "0", 
            "-dim", "512", 
            "-thread", str(self.threads), 
            "-ws", "5", 
            "-neg", "10", 
            "-input", lang_bpe_all, 
            "-output", lang_bpe_all
        )

        self.workflow.add_jobs(fasttext)
        self.logger.info("Cross-lingual embeddings (pre-training results) in: {0}".format(bpe_vec))

        if self.training:
            ###########################################################################
            ################################ Training #################################
            ###########################################################################

            MONO_DATASET = "'{0}:{1},,;{2}:{3},,'".format(
                LANGS[0], 
                lang_binarized[LANGS[0]], 
                LANGS[1], 
                lang_binarized[LANGS[1]]
                )

            PARA_DATASET = "'{0}:,{1},{2}'".format(
                '-'.join(LANGS), 
                corrected_valid, 
                corrected_test
                )

            PRETRAINED = bpe_vec # Pretrained model produced by fasttext task

            training = Job("training")
            training_out = File("trained-{0}-{1}.out".format(LANGS[0], LANGS[1]))

            training.add_inputs(bpe_vec)
            for lang in LANGS:
                training.add_inputs(lang_binarized[lang], file_binarize_valid[lang], file_binarize_test[lang])

            training.set_stdout(training_out)
            training.add_args(
                '--exp_name', str(RUN_ID), 
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


            training.add_condor_profile(request_cpus=self.threads)
            training.add_condor_profile(request_gpus=self.gpus)
            self.workflow.add_jobs(training)
            self.logger.info("Model trained => {0}".format(training_out))

            ###########################################################################
            #############################  End Training ###############################
            ###########################################################################

        self.logger.info("Creating workflow done.")

    def run(self, submit=False):
        try:
            self.workflow.plan(
                    dir=str(WORK_DIR),
                    relative_dir=RUN_ID,
                    output_sites=["local"],
                    input_dirs=[str(MONO_PATH), str(PARA_PATH)],
                    output_dir=str(TOP_DIR / "output"),
                    cleanup="leaf",
                    submit=submit
            )#.wait()
        except Exception as e:
            print(e)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--cores", type=int, default=1, help="Number of threads")
    parser.add_argument("-g", "--gpus", type=int, default=1, help="Number of gpus")
    parser.add_argument("-s", "--singularity", action="store_true", help="Use singularity containers")
    parser.add_argument("-t", "--training", action="store_true", help="Activate training (require NVIDIA GPU)")

    args = parser.parse_args()
    
    wf = WorkflowNLP(RUN_ID, threads=args.cores, gpus=args.gpus, is_singularity=args.singularity, training=args.training, logger=LOGGER)
    wf.set_properties()
    wf.set_sites()
    wf.set_transformations()
    wf.set_replicas()
    # To activate the training task set  training=True (REQUIRED GPU+CUDA)
    wf.set_jobs()

    wf.run(submit=False)

