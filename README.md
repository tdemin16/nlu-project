# Natural Language Understanding Project
In this repository you can find the source code for the Natural Language Understanding Project, the results of experiments, some movie reviews collected by me as well as the project report.
Please read the project's report before running the code.
The code is written in Python 3.9.7 and has not been tested with other versions.

## Trained weights
The trained weights are available [here](https://drive.google.com/drive/folders/10RVRNd8bQQB6rzsvvKX6oSMlxHTDNFw1?usp=sharing). Some weights are not there due to problems related to Colab Sessions. In some cases, Colab session dropped before I was able to store them. In any case, at least 1 set of weights is available for each setting.

## Environment
The code has been developed using a conda environment, so to set it up run:
```bash
$ conda env create -n nlu --file environment.yml
```

I have also exported the requirements for a pip virtual environment (but I recommend to use conda). So to create a pip virtual environment, run:
```bash
$ python3 -m venv nlu_env
$ source nlu_env/bin/activate
$ pip install requirements.txt
```
> Be sure to create a virtual environment with Python 3.9.7 otherwise it is not guaranteed to work.

## Settings
In order to simplify the execution, each training setting can be changed in `settings.py`. There you can find:
* Predefined paths for saving and the path for the root directory of the project. It is not recommended to change them.
* Cross-Validation settings. The most important one is `FOLD_N` which is used to change the fold number Neural Networks training.
* Weight decay coefficient for GRU.
* Batch sizes: change them according to your computational power. Default ones should suffice for training with Colab Free GPUs.
* Training Epochs.
* Learning Rates.
* GRU settings:
  * Embedding dimension;
  * Size of the hidden state;
  * Pad token;
  * Attention: This boolean settings allows turning off/on the use of attention in GRU.
* Filter setting: Selects whether to filter or not objective sentences. Based on the approach this will change its behavior:
  * With Naive Bayes, since both models are trained in the same run, it will use the trained subjectivity detector to filter out objective sentences;
  * With GRU, it will load the trained weights in subjectivity detection. These will be located in `weights/gru/subj.pth`. The weights' directory is automatically created after the subjectivity detector is trained;
  * With Transformers, it will load the filtered dataset in `weights/transformer/filt_data.pkl`. To create it run `$ python transformer/filter_sents.py`
* Device and Saving settings;

## Training
### Naive Bayes (1st baseline)
Both classifiers are trained by running 
```bash
$ python train_baseline.py
```
Depending on the `FILTER` setting, the polarity detector will be trained on a filtered or a not filtered dataset.
Both classifiers will be stored in `weights/baseline/`

### GRU
To train the subjectivity detector, use:
```bash
$ python gru/train_subj.py
```
While, for the polarity classifier, use:
```bash
$ python gru/train_pol.py
```
In order to **enable attention**, set `ATTENTION=True` in settings.\
To **filter out objective sentences**, set `FILTER=True` in settings.\
Weights are stored in `weights/gru` with a name composed of `task_model_fold.pth`. \
**Remove** "`_fold`" **to use the subjectivity weights when filtering out objective sentences** (e.g. `subj_cls_1.pth` $\rightarrow$ `subj_cls.pth`).
> To filter out objective sentences, make sure to train the subjectivity detector first, and then to train the polarity classifier (use `FILTER=True`), otherwise it won't work.\
> If `FILTER=True`, make sure the trained weights in subjectivity detection have been trained with the same `ATTENTION` setting as the current one (for polarity classification).
> Remove the "`_fold`" also to the w2id filename.

### Transformer
To train the subjectivity detector, use:
```bash
$ python transformer/train_subj.py
```
While, for the polarity classifier, use:
```bash
$ python transformer/train_pol.py
```
Since the filtering operation is much longer with transformers than GRU, filtering and training of the polarity classifier are separated.
First you need to filter out the objective sentences and create the filtered dataset, then you can train the polarity classifier with filtered data. \
To filter out the dataset, run:
```bash
$ python transformer/filter_sents.py
```
This will produce the filtered dataset, and it will store it in `weights/transformer/filter_data.pkl`.\
Be sure the subjectivity detector has been trained appropriately. Moreover, be sure to remove "`_fold`" as in GRU.\
To run the polarity classifier on filtered data, set `FILTER=True` in the settings.