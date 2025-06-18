
# LLM4Rec: Temporal Expression Format Exploration

This repository explores the **impact of different temporal expression formats** on large language model-based recommendation (LLM4Rec), based on a modified version of [LLM-SRec](https://github.com/Sein-Kim/LLM-SRec).

- **Model**: LLaMA-3.2-3b-instruct
- **Dataset**: [Amazon Review Data (2023)](https://amazon-reviews-2023.github.io/)

---

## 📦 Environment Setup

```bash
conda create -n [env_name] pip
conda activate [env_name]
pip install -r requirements.txt
```

---

## 🔧 Step 1: Pretrain CF Recommender (SASRec)

This step downloads the dataset automatically and trains a base CF model:

```bash
cd SeqRec/sasrec
python main.py --device 0 --dataset CDs_and_Vinyl
```

---

## 🚀 Step 2: Train LLM4Rec (Item Retrieval Task)

Trains the model and evaluates performance on the test set:
> **Note**: To train the **original LLM-SRec** (baseline), please refer to the original [LLM-SRec repository](https://github.com/Sein-Kim/LLM-SRec).
```bash
python main.py --device 0 \
  --train \
  --model_version B \           # B / C / D / E / F
  --rec_pre_trained_data CDs_and_Vinyl \
  --save_dir model_train
```

---

## 📊 Experimental Results (Amazon CDs)

| Model Version                 | NDCG@10 | NDCG@20 | HR@10  | HR@20  |
|------------------------------|---------|---------|--------|--------|
| Original (LLM-S)             | 0.3222  | 0.3632  | 0.5483 | 0.7111 |
| **B)** Hour + Seconds Format | 0.3537  | 0.3931  | 0.5839 | 0.7400 |
| **C)** Human-Readable Date   | 0.3792  | 0.4161  | 0.6077 | 0.7537 |
| **D)** Relative Time Format  | 0.3592  | 0.3974  | 0.5939 | 0.7446 |
| **E)** Session-Relative Time | 0.3506  | 0.3893  | 0.5782 | 0.7313 |
| **F)** Multi-Dimensional Time| 0.3780  | 0.4142  | 0.6036 | 0.7469 |

---

## 🧠 Temporal Format Examples

- **Original**  
  `User A watched 'Avengers' on 2024-01-26, then watched 'Spider-Man' on 2024-01-15.`

- **B) Absolute Timestamp Format**  
  `User A watched 'Avengers' on 2024-01-26T09:23:47Z, then watched 'Spider-Man' on 2024-01-15T10:15:22Z.`

- **C) Human-Readable Format**  
  `User A watched 'Avengers' on Jan 1, 2022 at 9:23 AM. Then watched 'Spider-Man' on Jan 10, 2022 at 10:15 AM.`

- **D) Time Interval Format**  
  `User A watched 'Movie A', then after 25 seconds watched 'Movie B', then after 5 minutes watched 'Movie C'.`

- **E) Session-Relative Format**  
  `t=0s: Movie A (session start) → t=25s: Movie B → t=341s: Movie C → t=3295s: Movie D`

- **F) Multi-Dimensional Time Format**  
  `Movie A @ 09:23:47 [morning_peak, weekday, work_hour, gap_from_last=0s]`

---

