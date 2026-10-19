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
| `model.tokenizer_args.pretrained_model_name_or_path` | `meta-llama/<model>` (gated) | `open-unlearning/tofu_<model>_full` | train + eval |
| accelerate `--config_file` | `default_config.yaml` | `t4_config.yaml` | train |

Everything else matches `scripts/tofu_unlearn.sh` and the experiment config: lr 1e-5, 10 epochs, 1 warmup epoch, weight decay 0.01, `paged_adamw_32bit`, 4 per device × 4 accumulation steps × 2 GPUs = 32 effective batch, gradient checkpointing, and method args from `configs/trainer/<method>.yaml`.

## Measured limits (Llama-3.2-1B, NPO, 2026-10-19)

- 1× T4, fp32 weights + fp16 autocast: **out of memory**. Train on 2 GPUs with DeepSpeed.
- Full TOFU eval on 1× T4 in fp16: peak about 13 GB.
- Kaggle quota: 30 GPU hours per week, 12-hour maximum per session.

## Known deviations from `docs/repro.md`

- Hardware and precision: 2× L40s with bf16 there, versus 2× T4 with fp16 here. Expect small numeric differences. fp16 loss scaling may also skip a few early steps on overflow.
- `docs/repro.md` lists "8 per device, 4 grad accum" for batch 32, but `scripts/tofu_unlearn.sh` uses 4 per device. This page follows the script.

## Repo quirks hit along the way (not changed)

- `src/trainer/unlearn/rmu.py` imports `deepspeed` unconditionally.
- `load_trainer_args` (`src/trainer/__init__.py`) divides by `torch.cuda.device_count()`, so it crashes on machines without CUDA unless `trainer.args.warmup_epochs=0`.
