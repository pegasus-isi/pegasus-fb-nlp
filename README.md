# pegasus-fb-nlp
Example of a Facebook NLP pipeline taken from here https://github.com/facebookresearch/UnsupervisedMT

![](img/fb-nlp.svg)

## Input data
If you are allowed an ISI machine, input archives for EN and FR can be found here `/nfs/v5/lpottier/ml-workflows/nlp/pegasus-fb-nlp/input/mono`
Otherwise if the _.gz_ are not found in `input/mono`, the worflow will have to download them from `http://www.statmt.org/wmt14/training-monolingual-news-crawl/`

## Run the workflow
You need an HTCondor pool configured and Pegasus installed, then just run `./run.sh`. 
This workflow uses a container to execute several jobs, the container can be pulled from `docker pull lpottier/fb-nlp:0.1`.
Note that, Pegasus will pull the container automatically.


