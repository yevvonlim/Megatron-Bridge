# Step-3.5-Flash

[StepFun Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash) is a sparse Mixture-of-Experts language model with 288 routed experts (top-8) and a shared expert, per-layer hybrid full/sliding attention, per-layer 4-cycle RoPE (with llama3 scaling on full layers only), zero-centered RMSNorm, per-head sigmoid attention output gates, asymmetric SwiGLU clamping on the last two layers, and 3 Multi-Token-Prediction heads.

## Variants

| Variant | Total params | Active params | Layers | Hidden | Experts (topk) | HF path |
|---|---|---|---|---|---|---|
| Step-3.5-Flash | ~321B | ~38B | 45 | 4096 | 288 (top-8) | `stepfun-ai/Step-3.5-Flash` |
| Step-3.5-Flash-Mini-3B-A0.27B (custom) | ~3.1B | ~270M | 24 | 1024 | 80 (top-4) | (built locally via `Step3p5ModelProviderMini3B`) |

The 321B published variant is too large to load on most user hardware — the `examples/` here target it for reference, while the recipe in `src/megatron/bridge/recipes/step3p5/step3p5_mini.py` targets the locally-instantiated `Mini3B` variant for from-scratch pretraining on 8x A100-40GB.

## Conversion (321B published checkpoint)

See `conversion.sh`. The published checkpoint is bf16, no quantization, so no dequant step is required.

## Inference

See `inference.sh`. Generation requires a GPU big enough to hold the full 321B model in bf16 (~640GB).

## Pretraining the Mini3B variant from scratch

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
    scripts/training/run_recipe.py \
    --recipe step3p5_mini_3b_pretrain_config \
    train.train_iters=1000 \
    train.global_batch_size=256 \
    dataset.blend='([/path/to/data.bin],[1.0])'
```

## Architectural notes (deviations from naive port)

1. **Uniform attention head count.** The published 321B uses 64 query heads on full-attention layers and 96 on sliding layers (`attention_other_setting`). Megatron-Core's `GPTModel` does not natively support per-layer heterogeneous head counts, and `AGENTS.md` forbids editing `3rdparty/Megatron-LM/`. The bridge raises if a config sets `attention_other_setting` to differ from the base. The Mini3B variant uses uniform 24/4 heads, preserving the 1.5x Q-expansion ratio of the original full layers.
2. **MTP HF-format export is one-way.** The published modeling code drops MTP keys via `_keys_to_ignore_on_load_unexpected = [r"model\.layers\.45\.*", r"model\.layers\.46\.*", r"model\.layers\.47\.*"]`. Megatron-trained MTP weights round-trip through the file format but cannot be loaded by the published `Step3p5ForCausalLM` class.
3. **256K context dropped for from-scratch pretraining.** The Mini3B recipe trains at `seq_length=4096`. llama3 scaling fields are kept in config so a later finetune can extend.
4. **Static rope (no dynamic NTK).** Long-context dynamic NTK extension is out of scope for the from-scratch pretraining milestone.

## Known limitations

- Per-layer different attention head counts (`attention_other_setting`) are explicitly rejected by the bridge.
- HF Hub may not expose a Step-3.5-Flash tokenizer; the recipe defaults to `NullTokenizer` with `vocab_size=128896` for parameter sizing — supply `--tokenizer-model` once a real BPE is available.
- MTP weight export is best-effort (HF-side ignores them).
- `norm_expert_weight` config field is dormant in the published modeling code; behavior is hardcoded ON via `renormalize=True` in MoE forward.
