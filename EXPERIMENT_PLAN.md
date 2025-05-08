# ArXiv Ergodic‑Component Scaling Experiments

## 0 Overview

We measure how a causal language model's **in‑context token‑level cross entropy (XE)** scales with the number **K** of statistically independent ergodic components when **total training tokens are held constant**.

> *Why "cross entropy" and not "NLL"?*  For one‑hot targets the two are numerically identical.  We adopt the cross‑entropy terminology throughout for clarity.

* **Components** = eleven arXiv sub‑categories:
  `cs.CV, cs.AI, cs.SY, cs.CE, cs.PL, cs.IT, cs.DS, cs.NE, math.AC, math.GR, math.ST`
  (mutually disjoint and balanced in `ccdv/arxiv-classification`).
* **Scaling values** K ∈ {1, 2, 4, 8, 11}.
* * **Model size** Pythia‑70 M (fixed for all runs).

---

## 1 Dataset pipeline

1. **Download & cache** `ccdv/arxiv-classification` (train/val/test).
2. **Tokenise** with pythia tokenizer.
3. **Store** Parquet rows: `paper_id · category · List[int] tokens` (size varies).
4. **Dynamic window loader**
   – On every `__getitem__`, choose a *component* uniformly, choose a paper uniformly within it, then take a random **100‑token** slice.
   – Uniform component sampling guarantees equal long‑run exposure without enforcing per‑batch balance.

---

## 2 Token‑budget rationale

A safe rule for scratch training is ≥ 5 tokens per parameter **per epoch**.

| Model       | Params | Budget / epoch | Effective tokens (12 epochs) | Tokens / param |
| ----------- | ------ | -------------- | ---------------------------- | -------------- |
| Pythia‑70 M |  70 M  | **100 M**      | 1.2 B                        | 17×            |

Even with 100‑token windows we comfortably exceed the 5× rule; one A100‑40 GB handles a run in <2 h.

---

## 3 Experimental grid

| Factor             | Values                |
| ------------------ | --------------------- |
| **K**              | 1 / 2 / 4 / 8 / 11    |
| **Model size**     | 70 M (fixed)          |
| **Random seed**    | 0 / 1 / 2             |
| **Window length**  | 100 tokens            |
| **Context length** | 128 (BOS + 100 + pad) |

Runs = 5 × 3 = **15 trainings** (≈ 23 A100‑hours).

---

## 4 Training protocol (per run)

* **Batch (global)** 256 sequences.
* **Optimiser** AdamW (β = 0.9/0.95, ε = 1e‑8).
* **LR schedule** peak 2 e‑4, warm‑up 1 k steps, cosine decay to 10 %.
* **Steps** 100 k (\~12 effective epochs at 100 M tokens).
* **Regularisation** dropout 0.1, weight‑decay 0.1.
* **Framework** HF Transformers + Accelerate (fp16).
* **Checkpointing** every 10 k, **retain 12 snapshots** (`--save_steps 10000 --save_total_limit 12`) so any stage can be evaluated later.

---

## 5 Evaluation protocol

We compute cross entropy two ways:

1. **Per‑component XE (for Δ‑curve)**
   Freeze the checkpoint, set `model.eval()`/`torch.no_grad()`, then:

   * For every active component *c*:

     1. Sample **1 024** unseen 100‑token windows from **test**.
     2. Compute cross entropy on tokens 1…99 (BOS ignored).
     3. Average → *XE₍c₎*.
   * Average across components → *XE₍K₎*.
   * Report *XE₍K₎*, perplexity `exp(XE)`, and **ΔXE(K) = XE₍K₎ − XE₍1₎**.

2. **Single‑component baseline**
   For each of the 11 components individually, evaluate a model **trained on K = 1** (that same component) using the same 1 024‑window protocol.  Store these as *XE₁(component)* for detailed per‑topic analysis.

All results saved to `results/<run_id>.jsonl`.

---

## 6 Control

| Control                  | Procedure                              | Expected outcome             |
| ------------------------ | -------------------------------------- | ---------------------------- |
| **Random‑label shuffle** | Permute component IDs before training. | ΔXE ≈ 0 if pipeline correct. |

---

## 7 Deliverables Deliverables

* **`fig/xe_vs_K.png`** – log₂ K vs cross entropy, ±1 σ.
* **Table 1** – XE & perplexity for each K (mean of 3 seeds).
* **Appendix tables** – controls & per‑component baselines.
* **Code & configs** – tagged release (`fetch_arxiv.py`, `WindowDataset`, `train_single_k.py`, `eval_xe.py`).
* **Repro guide** – one‑liner in root `README.md`.

---

*Last updated: 2025‑05‑07*
