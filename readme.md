## TSSE-DMM

### Introduction

code for *TSSM-DMM:Topic Modeling for Short Texts based on Topic Subdivision and Semantic Enhancement*

### requirement

python==3.6

gensim==3.8.3

nltk==3.3

numpy==1.15.2

pyltp==0.2.1

progressbar33==2.4

you also need to download ltp_data_v3.4.0 to get ltp model, and put them into directory *ltp_data_v3.40*.

### dataset

all datasets contain two files, including **docs** and **labels**.

***sogouCA***:sogou **Chinese** news dataset, including **28731** samples

***snippets***:searchsnippets from **English** web, including **12265** samples

### how to run

#### running command

in src/model dir

```python
python main.py 
  --has_pre_model=0 --pre_corpus_path=../../corpus_sogouCA/docs
  --t_corpus_path=../../corpus_sogouCA/docs --language=c --K=40 --num_iters=100
```

#### parameters

***language***: corpus language

***has_pre_model***: weather has pretrained model

***pre_corpus_path***: path of pretrained corpus if has pretrained model

***t_corpus_path***: path of corpus for current model to train

***vector_path***: path of word vector

***K***: number of topics

### result

After run the train command, we will get the train result under the dir of corpus path.

Example:

We set corpus path with ''*corpus_sogouCA*" and run code,

the result files will be generated in directory "*corpus_sogouCA/result_single*", including topWords file(*.topWords*), topic distribution file(*.theta*) and model file(*.dat*).

