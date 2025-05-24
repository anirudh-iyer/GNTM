known problems: 1. cant use cuda.
2. heavily dependent on optimization script (optuna tuner)
3. Correlation label wont work if all your topics are labeled similarly.

# Strategic Next Steps
â Validate topic content
Go beyond top-10 words: sample representative documents for each topic.

Manually label if possible: does topic 9 actually represent a hate theme?

ð Introduce neutral or benign seeds
If all topics map to hate, introduce non-hate seed topics to distinguish better.

ð Use entropy/confidence to re-rank posts
Use high-confidence, low-entropy docs to build a clean training set for classifiers.

Use high-entropy posts for human review or multi-label models.

ð§  Plot Topic Co-occurrence Matrix
Try plotting cosine similarity or Î¸.T @ Î¸ to understand topic-topic similarity across documents.

If you'd like:

A cosine similarity heatmap

A table of "top N example posts per topic"

Or converting any of this into a structured report / notebook

# ? GNTM: Graph-based Neural Topic Model (Modified Version)

This repository contains a **modified version of GNTM** (Graph-based Neural Topic Model) with the following key enhancements:

- ? Seed-guided topic regularization
- ? Custom evaluation metrics: Topic Diversity, Seed Topic Document Coverage
- ? Structured logging and visualizations
- ? Enhanced data preprocessing and configurability

---

## ? Key Additions

### ? Seed-Guided Topic Regularization

Inject domain knowledge using **seed words** to guide topic learning. Each topic is associated with a set of user-defined seed tokens, and the model is regularized to align with these seeds.

? **Seed File Format** (Excel `.xlsx`)
```plaintext
token     topic
--------- -------
democracy politics
election  politics
````

? **Behavior**

* Tokens mapped to vocab indices
* Regularization term added to loss
* Increases weight on ? (topic-word distribution) for seeded words

---

### ? Additional Evaluation Metrics

#### ? Topic Diversity

Measures distinctiveness of top-k words per topic:

```python
Diversity@k = Unique(top-k words across all topics) / (k * num_topics)
```

#### ? Seed Topic Document Coverage

Number of documents strongly associated with each seeded topic.

? **Sample Output**

```text
Topic diversity is: 0.900 in top 5 words
...
Seed Topic 0: 35 documents
Seed Topic 1: 41 documents
...
```

---

## ? Data Preparation

Each dataset must include:

* `overall.csv` or `overall_stop.csv` with columns:

  * `content`, `label`, `train` (1=train, 0=test, -1=val)
* Word embeddings: `300d_words.npy`
* Optional: `seed_words.xlsx` in `datasets/YourDataset/`

? **Directory Structure**

```
datasets/
??? Instagram/
    ??? overall.csv
    ??? 300d_words.npy
    ??? seed_words.xlsx
```

---

## ? Run Instructions

### ? Train the Model

```bash
python main.py \
  --dataset Instagram \
  --model_type GDGNNMODEL \
  --num_topic 10 \
  --num_epoch 10 \
  --batch_size 150 \
  --learning_rate 0.0003 \
  --ni 300 \
  --nw 300 \
  --nwindow 5 \
  --STOPWORD \
  --device cpu
```

### ? Evaluate the Model

```bash
python main.py \
  --dataset Instagram \
  --model_type GDGNNMODEL \
  --eval \
  --load_path models/Instagram/.../model.pt \
  --device cpu
```

---

## ? Output Artifacts

* `log.txt`: Logs with training/validation/test losses
* `model.pt`: Trained model
* `whole_edge.csv`: Edge index weights
* `beta_edge_True_*.csv` / `beta_edge_False_*.csv`: Seed-related topic-pair distributions
* Visualizations and printed topic-word evaluations

---

## ? Acknowledgements

Original GNTM: [SmilesDZgk/GNTM](https://github.com/SmilesDZgk/GNTM)
Modified by: Anirudh Iyer and team for seed-guided topic modeling

---

## ? License

MIT License
