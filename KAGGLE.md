# Running on Kaggle (2× Tesla T4)

Kaggle's free GPU is 2× T4 (16 GB each, compute capability 7.5). The T4 supports neither bf16 nor FlashAttention-2, and the repo defaults assume both. This page lists every change from the defaults needed to run here.

```bash
bash scripts/kaggle_t4.sh NPO forget10 holdout10 retain90   # 2-GPU unlearn, then 1-GPU eval
```

## Environment

| Step | Default (README) | Kaggle | Why |
|---|---|---|---|
| Install | `pip install ".[lm-eval]"` + flash-attn | `pip install -r requirements.txt torchvision==0.24.1 lm-eval==0.4.11` | The Kaggle image ships torchvision built for torch 2.11, so the pinned torch 2.9.1 fails with `torchvision::nms does not exist`. `lm_eval` is imported unconditionally by `src/evals/lm_eval.py`. flash-attn is unusable on sm75. |
| Data | `python setup_data.py --eval_logs` | same | Provides the retain-model logs needed for `forget_quality`. |

## New files

| File | Change from the file it copies |
|---|---|
| `configs/accelerate/zero_stage3_fp16_config.json` | Copy of `zero_stage3_offload_config.json` with `"bf16": {"enabled": true}` replaced by `"fp16": {"enabled": "auto"}`. |
| `configs/accelerate/t4_config.yaml` | Copy of `default_config.yaml`, pointing to the fp16 DeepSpeed config. |
| `scripts/kaggle_t4.sh` | One `(model, split, trainer)` iteration of `scripts/tofu_unlearn.sh`, with the overrides below. |

## Overrides compared with `scripts/tofu_unlearn.sh`

| Override | Default | Kaggle | Applies to |
|---|---|---|---|
| `model.model_args.attn_implementation` | `flash_attention_2` | `sdpa` | train + eval |
| `model.model_args.torch_dtype` | `bfloat16` | `float32` for train (DeepSpeed keeps fp16 params + fp32 master weights), `float16` for eval | train / eval |
| `trainer.args.bf16` / `bf16_full_eval` | `True` / `True` | `False` / `False` | train |
| `trainer.args.fp16` | unset | `True` (dynamic loss scaling) | train |
| `model.tokenizer_args.pretrained_model_name_or_path` | `meta-llama/<model>` (gated) | `open-unlearning/tofu_<model>_full`, with no token needed ([proven identical](#tokenizer-equivalence-llama-32-1b-instruct)) | train + eval |
| accelerate `--config_file` | `default_config.yaml` | `t4_config.yaml` | train |

Everything else matches `scripts/tofu_unlearn.sh` and the experiment config: lr 1e-5, 10 epochs, 1 warmup epoch, weight decay 0.01, `paged_adamw_32bit`, 4 per device × 4 accumulation steps × 2 GPUs = 32 effective batch, gradient checkpointing, and method args from `configs/trainer/<method>.yaml`.

## Measured limits (Llama-3.2-1B, NPO, 2026-10-19)

- 1× T4, fp32 weights + fp16 autocast: **out of memory**. Train on 2 GPUs with DeepSpeed.
- Full TOFU eval on 1× T4 in fp16: peak about 13 GB.
- Kaggle quota: 30 GPU hours per week, 12-hour maximum per session.
- TOFU forget10 NPO (130 steps) takes about 50 min end to end: about 35 min of training at about 16 s/step, plus setup and eval. Both GPUs peak at about 14.9 GB as reported by `nvidia-smi`, which includes PyTorch's allocator cache. DeepSpeed logs "allocator cache flushes" warnings, which means memory pressure.

## Tokenizer equivalence (Llama-3.2-1B-Instruct)

Compared `meta-llama/Llama-3.2-1B-Instruct` with `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`, both loaded through the repo's `get_tokenizer` so the runtime pad/eos logic applies. Every TOFU example was encoded with the repo's own `preprocess_chat_instance`, using `template_args` from `configs/model/Llama-3.2-1B-Instruct.yaml`. The inputs were each question and paraphrased question, crossed with each answer, paraphrased answer and perturbed answer, in both training and generation mode. All of `input_ids`, `labels` and `attention_mask` were compared.

| Check | Result |
|---|---|
| Vocabulary | identical (128,256) |
| BOS / EOS / pad ids | identical (128000 / 128009 / 128009) |
| `padding_side` | identical (right) |
| `chat_template` | identical |
| `forget10`, `forget10_perturbed`, `holdout10`, `retain90`, `retain_perturbed`, `real_authors_perturbed`, `world_facts_perturbed` | **0 of 32,936 encodings differ** |

So the tokenizer override doesn't change any model input. Other model sizes haven't been checked.

## Results: TOFU forget10, NPO, Llama-3.2-1B-Instruct

Run with `bash scripts/kaggle_t4.sh NPO forget10 holdout10 retain90` on 2026-10-19 (single seed, 0).

Compared with the NPO row of `docs/repro.md`:

| Metric | `docs/repro.md` | Kaggle 2× T4 fp16 | Gap |
|---|---|---|---|
| forget_quality | 0.02 | 0.030 | 1.5× (same order; this is a KS-test p-value) |
| model_utility | 0.46 | 0.523 | **+0.06** |
| forget_truth_ratio | 0.70 | 0.673 | −0.03 |

Metrics that `docs/repro.md` doesn't report, compared with the reference logs from `setup_data.py --eval_logs`:

| Metric | Fine-tuned (`full`) | Kaggle NPO | Retain (`retain90`) |
|---|---|---|---|
| forget_Q_A_Prob | 0.880 | 0.101 | 0.116 |
| forget_Q_A_ROUGE | 0.820 | 0.332 | 0.379 |
| extraction_strength | 0.706 | 0.058 | 0.059 |
| privleak | −99.5 | 5.9 | 23.5 |

forget_quality and forget_truth_ratio match. model_utility is 0.06 higher than the reference. With one run, and no error bars in `docs/repro.md`, that gap can't be attributed to noise, fp16, or the batch-size ambiguity (see Known deviations below).

## Known deviations from `docs/repro.md`

- Hardware and precision: 2× L40s with bf16 there, versus 2× T4 with fp16 here. Expect small numeric differences. fp16 loss scaling may also skip a few early steps on overflow.
- `docs/repro.md` lists "8 per device, 4 grad accum" for batch 32, but `scripts/tofu_unlearn.sh` uses 4 per device. This page follows the script.

## Repo quirks hit along the way (not changed)

- `src/trainer/unlearn/rmu.py` imports `deepspeed` unconditionally.
- `load_trainer_args` (`src/trainer/__init__.py`) divides by `torch.cuda.device_count()`, so it crashes on machines without CUDA unless `trainer.args.warmup_epochs=0`.
