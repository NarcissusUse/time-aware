## Overview

This repository is designed for exploring the Impact of Different Temporal Expression Formats on LLM4Rec. which modified the llm-srec repository (https://github.com/Sein-Kim/LLM-SRec)

- We use LLaMA-3.2-3b-instruct.

## Env Setting
```
conda create -n [env name] pip
conda activate [env name]
pip install -r requirements.txt
```

## Pre-train CF-RecSys (SASRec)

The data ([Amazon 2023](https://amazon-reviews-2023.github.io/)) is automatically downloaded when using the SASRec training code provided below.

```
cd SeqRec/sasrec
python main.py --device 0 --dataset CDs_and_Vinyl
```

## Train - Item Retrieval
The model saves when the best validation score is reached during training and performs inference on the test set.

```
python main.py --device 0 --train --rec_pre_trained_data CDs_and_Vinyl --save_dir model_train
```
