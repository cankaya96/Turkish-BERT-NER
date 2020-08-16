# Turkish-BERT-NER

I wanted to show how you can use your dataset for NER  with Turkish BERT
I’m making the assumption that the readers already have background information on the following subjects:\
1 - Named Entity Recognition (NER).\
2 - Bidirectional Encoder Representations from Transformers (BERT).\
3 - HuggingFace (transformers) Python library.

## HuggingFace Trainer Class

Transformers new Trainer class provides an easy way of fine-tuning transformer models for known tasks such as CoNLL NER.

This class will take care of training/evaluation loops, logging, model saving …etc. 

Which makes switching to other transformers models very easy. For this purpose we will use another class TFNerDataset which handles loading and tokenization of the data.

## Datasets

The datasets which I used you can find [here](https://github.com/cankaya96/Turkish-BERT-NER) I used test.csv and train.csv but they are not very well about animal entity because it is not much like the other entities (LOCATION, ORGANIZATION, PERSON).

## Preprocessing
First, we need to use csv files to make dataframes
```Python
import pandas as pd
import csv

testCsv=pd.read_csv("/content/drive/My Drive/SoftTech/Datasets/BERT/test.csv")
trainCsv=pd.read_csv("/content/drive/My Drive/SoftTech/Datasets/BERT/train.csv")
train_tuples=[]
trainWords=trainCsv["words"].copy()
trainLabels=trainCsv["labels"].copy()
for i in range(len(trainWords)):
  train_tuples.append([i,trainWords[i],trainLabels[i]])



test_tuples=[]
testWords=testCsv["words"].copy()
testLabels=testCsv["labels"].copy()
for i in range(len(testWords)):
  test_tuples.append([i,testWords[i],testLabels[i]])


train_df = pd.DataFrame(train_tuples, columns=[ 'sentence_id','words', 'labels'])
test_df = pd.DataFrame(test_tuples, columns=['sentence_id','words', 'labels'])
train_df
```
After that we need to know how many labels we have in our dataset

```Python
# a list that has all possible labels 
labels = train_df['labels'].unique()
label_map =  {i: label for i, label in enumerate(labels)}
num_labels = len(labels)
print(labels) 
print(label_map) 
```
## TFNerDataset
This was actually NerDataset but it was changed and I found the new version in [here](https://github.com/huggingface/transformers/tree/master/examples/token-classification) we must install utils_ner.py, run_ner.py and tasks.py to continue.

```Python
!wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/token-classification/utils_ner.py

!wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/token-classification/run_ner.py

!wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/token-classification/tasks.py

```
The model we’re using is a cased base BERT model ([BERTurk](https://github.com/stefan-it/turkish-bert)) pre-trained on a Turkish corpus of size 35GB and 44,04,976,662 tokens.

```Python
model_args = dict()

# Path to pretrained model or model identifier from huggingface.co/models
model_args['model_name_or_path'] = 'dbmdz/bert-base-turkish-cased' 

# Where do you want to store the pretrained models downloaded from s3
model_args['cache_dir'] = "/content/drive/My Drive/SoftTech/Models"

# we skip basic white-space tokenization by passing do_basic_tokenize = False to the tokenizer
model_args['do_basic_tokenize'] = False


data_args = dict()

data_args['data_dir'] = "/content/drive/My Drive/SoftTech/Datasets/BERT/train.txt"

# "The maximum total input sequence length after tokenization. Sequences longer "
# "than this will be truncated, sequences shorter will be padded."
data_args['max_seq_length'] = 512 

# Overwrite the cached training and evaluation sets
# this means the model does not have to tokenize/preprocess and cache the data each time it's called
# this can be made different for each NerDataset (training NerDataset, testing NerDataset)
data_args['overwrite_cache'] = False
```

Here, I’m also defining AutoModelForTokenClassification which is basically our BERT model with a classification head on top for Token classification.

```Python
!pip install transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer
)

config = AutoConfig.from_pretrained(
    model_args['model_name_or_path'],
    num_labels=num_labels,
    id2label=label_map,
    label2id={label: i for i, label in enumerate(labels)},
    cache_dir=model_args['cache_dir']
)

# we skip basic white-space tokenization by passing do_basic_tokenize = False to the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_args['model_name_or_path'],
    cache_dir=model_args['cache_dir'],
    do_basic_tokenize = model_args['do_basic_tokenize']
)

model = AutoModelForTokenClassification.from_pretrained(
    model_args['model_name_or_path'],
    config=config,
    cache_dir=model_args['cache_dir']
)
```
Now we need to make some settings

```bash
!export OUTPUT_DIR=germeval-model
!export BATCH_SIZE=32
!export NUM_EPOCHS=3
!export SAVE_STEPS=750
!export SEED=1
```
Now we can import seqeval

```bash
!pip install seqeval
```
Now if you use Google Colab you can add the next code to use GPU

```Pyhon
import os, sys, shutil
import time
import gc
from contextlib import contextmanager
from pathlib import Path
import random
import numpy as np, pandas as pd
from tqdm import tqdm, tqdm_notebook

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

USE_APEX = True

if USE_APEX:
            with timer('install Nvidia apex'):
                # Installing Nvidia Apex
                os.system('git clone https://github.com/NVIDIA/apex; cd apex; pip install -v --no-cache-dir' + 
                          ' --global-option="--cpp_ext" --global-option="--cuda_ext" ./')
                os.system('rm -rf apex/.git') # too many files, Kaggle fails
                from apex import amp
```

after that we need a JSON file to configuration 

```
{
    "data_dir": "/content/drive/My Drive/SoftTech/Datasets/BERT",
    "labels": "/content/drive/My Drive/SoftTech/Datasets/BERT/labels.txt",
    "model_name_or_path": "dbmdz/bert-base-turkish-cased",
    "output_dir": "/content/drive/My Drive/SoftTech/Models/Model_output/",
    "max_seq_length": 128,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 32,
    "save_steps": 750,
    "seed": 1,
    "do_train": true,
    "do_eval": true,
    "do_predict": true,
	"fp16":true
}
```
if you don't use GPU you don't need to use fp16. this JSON file must be added to your Drive to give access from your colab page.

At the end you can use run_ner.py and config.json to train and make prediction from your datasets.


```Python
from importlib import import_module
!pip install tasks
!pip install conllu

module = import_module("tasks")
#token_classification_task=getattr(module, model_args)
!python3 "/content/run_ner.py" "/content/config.json"
```
Now you can find your test_results.txt and test_predictions.txt files in the path that you wrote in your JOSN files output_dir.


