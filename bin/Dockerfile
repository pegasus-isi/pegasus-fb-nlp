# Use an official Ubuntu as a parent image
#FROM ubuntu:18.04
FROM nvcr.io/nvidia/pytorch:18.04-py3

# Set the working directory to /
WORKDIR /

RUN apt update && apt install -y --no-install-recommends wget gzip curl python3 python3-pip python3-setuptools git gcc build-essential default-jdk ant
RUN pip3 install --upgrade pip
RUN ln -fs /usr/bin/python3 /usr/bin/python

COPY requirements.txt tokenize.sh concatenate.sh concat-bpe.sh preprocess.sh tokenize-validation.sh /

RUN git clone https://github.com/lpottier/UnsupervisedMT.git && \
	cd UnsupervisedMT/NMT/ && \
	mkdir tools && \
	cd tools && \
	git clone https://github.com/moses-smt/mosesdecoder.git && \
    git clone https://github.com/glample/fastBPE && \
    git clone https://github.com/facebookresearch/fastText.git && \
	cd fastBPE && \
	g++ -std=c++11 -pthread -O3 fastBPE/main.cc -o fast && \
	cd ../fastText/ && \
	make

# Install any needed packages specified in requirements.txt
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

RUN rm -rf /var/lib/apt/lists/*
