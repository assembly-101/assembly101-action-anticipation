# TempAgg Action Anticipation baseline for Assembly101

PyTorch implementation of [TempAgg](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610154.pdf) used for the Action Anticipation task in the paper:

F. Sener et al. "**Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities**", CVPR 2022 [[paper](https://arxiv.org/pdf/2203.14712.pdf)]

This repository has been adapted from the original [TempAgg repository](https://github.com/dibschat/tempAgg) to train and evaluate on the Assembly101 dataset.

## Contents
* [Dependencies](#dependencies)
* [Overview](#overview)
* [Annotations](#annotations)
* [Features](#features)
* [Evaluate](#evaluate)
  * [Validation](#validation)
  * [Test](#test)
* [Train](#train)


## Dependencies
* Python3
* PyTorch
* Numpy, Pandas, PIL
* lmdb, tqdm

## Overview

This repository provides code to train, validate and generate test predictions (preds.json) for the task of action anticipation.

## Annotations

The action anticipation annotations and can be found [here](https://drive.google.com/drive/folders/1i_JsDmFt_sQ1T5ohEPAkCyyUkrJnY-rL). For anticipation, a subset of the verbs (17) from the original 24 verbs have been selected since verbs like `"attempt to"` are infeasible for anticipation.

## Features

TempAgg requires per-frame features as input. TSM (8-frame input) has been used for extracting 2048-D per-frame features which can be downloaded from our [Gdrive](https://drive.google.com/drive/folders/1nh8PHwEw04zxkkkKlfm4fsR3IPEDvLKj). Please follow [this](https://github.com/assembly-101/assembly101-download-scripts) for requesting drive access to download the `.lmdb` TSM features.

## Pretrained Models

Will be released soon.

## Evaluate

### Validation
<u>**Top-5 recall (T5R) results on the validation set for different splits**</u>
|  Split      | Verb T5R (%) | Object T5R (%) | Action T5R (%) |
|:-----------:|:------------:|:--------------:|:--------------:|
| **Overall** |     59.11    |      26.27     |      8.53      |
|  **Tail**   |     53.10    |      25.93     |      3.94      |
| **Unseen**  |     58.77    |      23.00     |      8.34      |

**Note**: The above result is produced after avg-pooling scores from all views.

To evaluate our model on the validation set (for all views fixed+ego), run the following:

```bash
python main.py --mode validate \
      --path_to_data "path_to_data" \
      --path_to_models "path_to_models" \
      --path_to_anno "path_to_CSV_annotations" \
      --modality fixed+ego \
      --views all \
      --batch_size 32 --num_workers 6
```

The above command will provide accuracy and recall only on the overall split. To get the Top-5 Recall on Tail and Unseen splits, run the above command with `--save_json "path_to_json"`. This will generate a `preds.json` file. Then running `evaluate.py` will provide results separately on the overall, tail and unseen splits.

`evaluate.py` requires 2 command line arguments.
```bash
python evaluate.py "path_to_GT.json" "path_to_preds.json"
```

In order for the `evaluate.py` to not throw any error, please maintain the same file structure of `data/` as provided [here](https://drive.google.com/drive/folders/1i_JsDmFt_sQ1T5ohEPAkCyyUkrJnY-rL).

### Test
To get similar overall, tail and unseen split results on the test set, run the following to generate `preds.json` for the test set:

```bash
python main.py --mode test \
      --path_to_data "path_to_data" \
      --path_to_models "path_to_models" \
      --path_to_anno "path_to_CSV_annotations" \
      --modality fixed+ego \
      --views all --save_json data \
      --batch_size 32 --num_workers 6 \
      --model_name_ev 'name of the .pth.tar checkpoint file'
```
The test set labels have been withheld for challenge purposes. So please submit the `preds.json` to our [Codalab Action Anticipation challenge](TBD) to produce Top-5 Recall results on Overall/Tail/Unseen splits separately.

## Train

To train the model, run the following:

```bash
python main.py --mode train --epochs 15 \
      --path_to_data "path_to_data" \
      --path_to_models "path_to_models" \
      --path_to_anno "path_to_CSV_annotations" \
      --modality fixed+ego \
      --views all --past_attention \
      --batch_size 32 --num_workers 6
```

add `--trainval` to the command if you want to train on both training + validation sets.
