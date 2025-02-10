## LLM-SRec: For sequential recommendation

This repository is designed for implementing LLM-SRec.

- We use LLaMA-3.2-3b-instruct.

## Env Setting
```
conda create -n [env name] pip
conda activate [env name]
pip install -r requirements.txt
```

## Pre-train CF-RecSys (SASRec)


```
cd SeqRec/sasrec
python main.py --device 0 --dataset Industrial_and_Scientific
```


## Train - Item Retrieval


```
python main.py --device 0 --train --rec_pre_trained_data Industrial_and_Scientific --save_dir model_train --batch_size 20
```
