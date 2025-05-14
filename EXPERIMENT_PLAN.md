# ArXiv Ergodic‑Component Scaling Experiments

## 0 Overview

We measure how a causal language model's **in‑context token‑level cross entropy (XE)** scales with the number **K** of statistically independent ergodic components when **total training tokens are held constant**. That means that as we add more components, we need to reduce the number of tokens per component to keep the total number of tokens constant.

> *Why "cross entropy" and not "NLL"?*  For one‑hot targets the two are numerically identical.  We adopt the cross‑entropy terminology throughout for clarity.

* **Components** = eleven arXiv sub‑categories:
  `cs.CV, cs.AI, cs.SY, cs.CE, cs.PL, cs.IT, cs.DS, cs.NE, math.AC, math.GR, math.ST`
  A major task is to analyze the dataset and figure out how to structure this. My main idea right now is to use the categories with the top five most data in their training set. Then we train first on the smallest of those, then for the next one we train on smallest/2, and next largest truncated to smallest/2, etc.
* **Scaling values** A major task is to determine this!
* * **Model size** A major task is to determine this!
* Creating the dataset is a major task. It should go something like this, this is just a rough idea, please rethink on your own!. Also keep in mind that we want to use variable amounts of data from each set, as described above, for different experiments.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", use_fast=True)

from datasets import load_dataset
from collections import Counter

stream = load_dataset("ccdv/arxiv-classification",
                      "no_ref",
                      split="train",
                      streaming=True)

counts = Counter()
for ex in stream:
    counts[ex["label"]] += 1

top5_ids = [idx for idx, _ in counts.most_common(5)]
print("Top-5 label IDs:", top5_ids)

train = load_dataset("ccdv/arxiv-classification",
                     "no_ref",
                     split="train") \
        .filter(lambda ex: ex["label"] in top5_ids)

def tok_map(batch):
    out = tok(batch["text"],
              add_special_tokens=False,
              truncation=False)
    # keep the label
    return {"input_ids": out["input_ids"],
            "label": batch["label"]}

tokd = train.map(tok_map,
                 batched=True,
                 remove_columns=train.column_names,
                 num_proc=8)   # use your cores

def chunk(batch):
    flat = sum(batch["input_ids"], [])          # concat
    n = len(flat) // 100
    flat = flat[:n * 100]
    chunks = [flat[i*100:(i+1)*100] for i in range(n)]
    # broadcast label over all its chunks
    labels = [batch["label"][0]] * n
    return {"input_ids": chunks, "label": labels}

chunked = tokd.map(chunk,
                   batched=True,
                   batch_size=1024,
                   remove_columns=tokd.column_names,
                   num_proc=8)

chunked.save_to_disk("arxiv_top5_len100_pythia")

from datasets import load_from_disk
ds = load_from_disk("arxiv_top5_len100_pythia")
ds.set_format("torch")     # or "numpy"


```

---

## 1 Dataset pipeline

1. **Download & cache** `ccdv/arxiv-classification` (train/val/test).
2. **Tokenise** have to decide on a tokenizer.
3. **Store** have to think through and decide on this
4. **Dynamic window loader**
   – On every `__getitem__`, we want a 100-token randomly selected window from the training set. A single epoch should go through the entire training set once. And we want to go have 3-5 epochs (to be deterimed).


---

## 2 Token‑budget rationale

Think through this.