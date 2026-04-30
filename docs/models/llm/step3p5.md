# Step-3.5-Flash

[StepFun Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash) is a sparse Mixture-of-Experts language model with several non-standard architectural pieces that the bridge captures via custom modules in `src/megatron/bridge/models/step3p5/modules.py`.

## Supported Variants

| Variant | Total params | Active params | HF path / source |
|---|---|---|---|
| Step-3.5-Flash | ~321B | ~38B | `stepfun-ai/Step-3.5-Flash` |
| Step-3.5-Flash-Mini-3B-A0.27B | ~3.1B | ~270M | `Step3p5ModelProviderMini3B` (locally instantiated) |

## Architecture summary

| Feature | Megatron flag / module |
|---|---|
| Sparse MoE w/ sigmoid + bias routing, fp32 gate | `moe_router_score_function="sigmoid"`, `moe_router_enable_expert_bias=True`, `moe_router_dtype="fp32"`, `moe_router_pre_softmax=False` |
| Frozen router bias (HF `requires_grad=False`) | `moe_router_bias_update_rate=0.0` |
| Topk renormalize + scaling factor 3.0 | `moe_router_topk_scaling_factor=3.0` |
| Shared expert (parallel summation) | `moe_shared_expert_intermediate_size`, `moe_shared_expert_overlap=True` |
| Per-layer dense / MoE | `moe_layer_freq=[0]*k+[1]*(N-k)` |
| Multi-Token Prediction (3 heads) | `mtp_num_layers=3` |
| Hybrid full + sliding-window attention | `Step3p5TEDotProductAttention` (per-layer `window_size`) |
| Per-layer 4-cycle RoPE (theta + partial-rotary) | `Step3p5RotaryEmbedding` (slot 0 = full-attn = llama3 + half-rope) |
| llama3 RoPE scaling on full layers only | `yarn_only_full_attention=True`; manual llama3 init (MCore stock has wrong coefficients) |
| Zero-centered RMSNorm (`x*(w+1)`) | `layernorm_zero_centered_gamma=True` |
| QK norm (per-head) | `qk_layernorm=True` |
| Per-head sigmoid output gate | `Step3p5SelfAttention.head_gate` (column-parallel) |
| Asymmetric SwiGLU clamp at last 2 layers | `Step3p5MLP` runtime-aware (gate one-sided, up two-sided) |

## Conversion (321B published checkpoint)

```bash
HF_MODEL=stepfun-ai/Step-3.5-Flash
WORKSPACE=/workspace

uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model "${HF_MODEL}" \
    --megatron-path "${WORKSPACE}/Step-3.5-Flash" \
    --torch-dtype bfloat16 \
    --trust-remote-code
```

Note: ~640GB in bf16. Requires multi-node H100/H200.

## Pretraining (Mini3B variant from scratch on 8x A100-40GB)

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
    scripts/training/run_recipe.py \
    --recipe step3p5_mini_3b_pretrain_config \
    train.train_iters=100000 \
    train.global_batch_size=256 \
    dataset.blend='([/path/to/data.bin],[1.0])'
```

Default parallelism: TP=2, PP=1, EP=4, sequence_parallel=True, seq_length=4096.

## Known limitations

- **Per-layer different attention head counts** are explicitly rejected by the bridge. Step-3.5's published config sets `num_attention_heads=64` for full layers and `attention_other_setting.num_attention_heads=96` for sliding layers (KV heads constant at 8). Megatron-Core's `GPTModel` does not natively support per-layer heterogeneous head counts and `AGENTS.md` forbids editing `3rdparty/Megatron-LM/`. The Mini3B variant uses uniform 24/4 heads, preserving the 1.5x Q-expansion ratio of the original full layers.
- **MTP HF-format export is one-way.** The published modeling code drops MTP keys (`_keys_to_ignore_on_load_unexpected = [r"model\.layers\.4[5-7]\.*"]`). Megatron-trained MTP weights round-trip through the file format but cannot be loaded by the published `Step3p5ForCausalLM` class.
- **Long-context dynamic NTK is not supported.** The custom `Step3p5RotaryEmbedding` is static (no `@dynamic_rope_update`). Extending context post-pretraining requires adding the dynamic-rope codepath.
- **Tokenizer.** HF Hub may not expose a Step-3.5-Flash tokenizer; the recipe defaults to `NullTokenizer` with `vocab_size=128896`. Supply `--tokenizer-model` once a real BPE is available.
- **`norm_expert_weight` config field is dormant** in the published modeling code; routing renormalization is hardcoded ON via the `renormalize=True` kwarg in MoE forward.
- **`sink` and `attention_other_setting` config fields** are not consumed.
- **256K context** is dropped for from-scratch pretraining at the Mini3B scale; llama3 scaling fields are kept in config so a later finetune can extend.
