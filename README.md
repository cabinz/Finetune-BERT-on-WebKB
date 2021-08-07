# Finetune-BERT-on-WebKB

Project source of NCSU 2021 Summer Gears. 

See [here](https://cabinz.github.io/2021summergears/2021/08/07/bert-on-webkb.html) for experiment report in Chinese.



## Requirements

| Tools / Packages | Version | Installation                                                 |
| ---------------- | ------- | ------------------------------------------------------------ |
| Python           | 3.6     | [Miniconda (conda 4.9.2 recommended)](https://docs.conda.io/en/latest/miniconda.html) |
| pytorch          | 1.7.1   | [Pytorch (GPU version recommended)](https://pytorch.org/get-started/locally/) |
| transformers     | 4.8.2   | [Huggingface Transformers (pip install recommended)](https://huggingface.co/transformers/installation.html) |
| matplotlib       | 3.3.4   | for visualization                                            |

* Make sure your pytorch version matches the CUDA version of your machine for the program to run on GPU.
* Recommend installing transformers using pip instead of conda, since the tokenizer from conda installation may require extra `GLIBC_2.29` from your machine system.



## Dataset

[4 Universities Data Set](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/) from [WebKB](https://www.cs.cmu.edu/~webkb/) is used for fine-tuning.

The full texts of the web pages are pre-processed by removing all the noninformative texts including HTML elements, front matters and punctuations.



## Start-up

The shell script for starting the training directly is in [run.sh](https://github.com/cabinz/Finetune-BERT-on-WebKB/blob/main/run.sh). 

Huggingface's transformers package is used for implementing and loading BERT. Internet connection is required for downloading the pre-trained model and the tokenizer from huggingface when running the scripts. You can also cache the model parameters and tokenizer on your machine for local running using methods provided by huggingface.

Several hyperparameters and settings are required including:

| Parameters      | Type    | Description                | Default                        |
| --------------- | ------- | -------------------------- | ------------------------------ |
| `--bert_name`   | string  | BERT name for classifier   | `bert-base-uncased`            |
| `--max_len`     | integer | max length of tokenization | `512`                          |
| `--batch_siz`   | integer | batch size                 | `16`                           |
| `--lr`          | float   | initial learning rate      | `5e-6`                         |
| `--num_epoch`   | integer | max number of epochs       | `100`                          |
| `--random_seed` | integer | global random seed         | `2021`                         |
| `--uni_lt`      | strings | list of university names   | all the five university labels |

`--bert_name` is for the `BertModel.from_pretrained()` from transformers package, which can be `bert-base-uncased` (by default) or `prajjwal1/bert-tiny` for BERT Tiny.



## Code Structure

* The model `BertClassifier` (BERT+Linear) and functions `train_loop` and `test_loop` for training and evaluation respectively are defined in [bertclf.py](https://github.com/cabinz/Finetune-BERT-on-WebKB/blob/main/bertclf.py) .

* The training and evaluating scripts are in the same file named [bertclf-train.py](https://github.com/cabinz/Finetune-BERT-on-WebKB/blob/main/bertclf-train.py) since the whole task is simple.

  The dataset splitting statistics, fine-tuned model and the visualization figures will be saved after training.

* Some helpful tool functions are defined in several files in [utils](https://github.com/cabinz/Finetune-BERT-on-WebKB/tree/main/utils) directory, including [dataloader.py](https://github.com/cabinz/Finetune-BERT-on-WebKB/blob/main/utils/dataloader.py)  module and [vis.py](https://github.com/cabinz/Finetune-BERT-on-WebKB/blob/main/utils/vis.py) module for data loading and visualization respectively.

* [dataset.tsv](https://github.com/cabinz/Finetune-BERT-on-WebKB/blob/main/dataset.tsv) is the pre-processed dataset that can directly loaded for training. 
  The function of data cleaning for the original dataset is in [utils/preprocessor.py](https://github.com/cabinz/Finetune-BERT-on-WebKB/blob/main/utils/preprocessor.py), with which you can customize your data cleaning by modifying the pre-processing functions in it. Make sure you've unzip the `webkb` directory of the original dataset to the root directory of the project for the pre-processing function.

* Directories [bertclf-base-save](https://github.com/cabinz/Finetune-BERT-on-WebKB/tree/main/bertclf-base-save) and [bertclf-tiny-save](https://github.com/cabinz/Finetune-BERT-on-WebKB/tree/main/bertclf-tiny-save) are for saving the fine-tuning results, logs and models from different pre-trained version of BERT.



## Results

Four-category classification (*student*, *faculty*, *course* and *project*) is conducted on the whole dataset, i.e. with samples from all five university labels including `misc`. Class *staff* and *department* are removed because of imbalance. (You can also customize your classification categories in [bertclf_train.py](https://github.com/cabinz/Finetune-BERT-on-WebKB/blob/main/bertclf-train.py))

With default hyperparameters and `Adam` as optimizer:

| Model     | n_epoch | Accuracy (%) |
| --------- | ------- | ------------ |
| BERT-Base | 10      | 97.17        |
| BERT-Tiny | 100     | 95.40        |
