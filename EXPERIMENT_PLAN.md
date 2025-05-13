# ArXiv Ergodic‑Component Scaling Experiments

## 0 Overview

We measure how a causal language model's **in‑context token‑level cross entropy (XE)** scales with the number **K** of statistically independent ergodic components when **total training tokens are held constant**. That means that as we add more components, we need to reduce the number of tokens per component to keep the total number of tokens constant.

> *Why "cross entropy" and not "NLL"?*  For one‑hot targets the two are numerically identical.  We adopt the cross‑entropy terminology throughout for clarity.

* **Components** = eleven arXiv sub‑categories:
  `cs.CV, cs.AI, cs.SY, cs.CE, cs.PL, cs.IT, cs.DS, cs.NE, math.AC, math.GR, math.ST`
  A major task is to analyze the dataset and figure out how to structure this. My main idea right now is to use the categories with the top five most data in their training set. Then we train first on the smallest of those, then for the next one we train on smallest/2, and next largest truncated to smallest/2, etc.
* **Scaling values** A major task is to determine this!
* * **Model size** A major task is to determine this!

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