# Speeding-Up-Transformers

Hands-on experiments with techniques for making transformers faster and cheaper to run and fine-tune, following *Hands-On Machine Learning* (HOMLP), Chapter 17.

## Contents

- **`speed-up-transformers.ipynb`** – walkthrough of speed-up techniques for decoder-only and encoder-decoder models:
  - Key/Value caching
  - Speculative decoding (OPT-350m target with OPT-125m draft)
  - Sparse attention (BigBird)
  - Approximate attention: LSH (angular) and Performer (FAVOR+)
  - Grouped-query attention
  - FlashAttention (blockwise implementation checked against `F.scaled_dot_product_attention`)
  - LoRA fine-tuning (GPT-Neo-125m)
  - Gradient checkpointing, gradient accumulation, multi-GPU parallelism (`DataParallel`)
- **`LoRA-Bert.ipynb`** – fine-tunes DistilBERT with LoRA (PEFT) for news topic classification on the `mdonigian/iab-news-classification` dataset.
- **`HOMLP_Chapter_17.pdf`** – the book chapter used as reference.
- **`LICENSE`** – license for the repo.

## LoRA-Bert details

- **Data:** `mdonigian/iab-news-classification` (35 IAB news categories), split into train / validation / test.
- **Model:** `distilbert-base-uncased` with a 35-class classification head, inputs truncated to 512 tokens.
- **LoRA (PEFT):** `r=16`, `alpha=32`, `dropout=0.1`, applied to the attention projections `q_lin` and `v_lin`; the `pre_classifier` and `classifier` heads are fully trained.
- **Training:** 3 epochs, batch size 8 with gradient accumulation 4, learning rate `2e-4`, best checkpoint loaded at the end (~37 min).
- **Result:** ~84.8% accuracy on the test set (10,628 examples, loss 0.436).

Training outputs (`lora_bert/`) are git-ignored.
