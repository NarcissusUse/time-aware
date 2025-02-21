## LLM-SRec: Lost in Sequence: Do Large Language Models Understand Sequential Recommendation?

This repository is designed for implementing LLM-SRec.

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
python main.py --device 0 --dataset Industrial_and_Scientific
```

We have also implemented LLM-SRec on the Gaudi-v2 environment (Note that `--nn_parameter` must be used for training models on Gaudi-v2):
```
python main.py --device 0 --dataset Industrial_and_Scientific --nn_parameter
```

## Train - Item Retrieval
The model saves when the best validation score is reached during training and performs inference on the test set.

```
python main.py --device 0 --train --rec_pre_trained_data Industrial_and_Scientific --save_dir model_train --batch_size 20
```

For Gaudi-v2:
```
python main.py --device 0 --train --rec_pre_trained_data Industrial_and_Scientific --save_dir model_train --batch_size 20 --nn_parameter
```