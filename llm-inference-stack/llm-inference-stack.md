# LLM Inference Stack — From Training to Serving

*Created: 2026-03-12*

## The Full Pipeline

```
1. Pre-trained model (Meta, DeepSeek, etc. spent millions training from scratch)
         │
         ▼
2. Unsloth — fine-tune on your data (backward pass: gradients, weight updates)
         │
         ▼
3. Export weights (safetensors, GGUF, etc.)
         │
         ▼
4. SGLang / vLLM / TensorRT-LLM — serve to users (forward pass only)
         │
         ▼
5. (Optional) NVIDIA Dynamo — orchestrate across a cluster of GPUs
```

Single GPU deployment skips Dynamo:
```
Unsloth → weights → SGLang or vLLM → users
```

Multi-node deployment:
```
Unsloth → weights → Dynamo → SGLang/vLLM across cluster → users
```

---

## Layer-by-Layer Breakdown

```
┌─────────────────────────────────────────────────┐
│           NVIDIA Dynamo (orchestration)          │  ← Distributed serving framework
│         Routes requests, manages clusters        │     (the "traffic controller")
├─────────────────────────────────────────────────┤
│     SGLang  │   vLLM   │  TensorRT-LLM          │  ← Inference engines
│            (the "cars")                          │     (scheduling, batching, API)
├─────────────────────────────────────────────────┤
│              FlashInfer (kernels)                 │  ← GPU kernel library
│            (the "engine parts")                  │     (raw GPU compute)
├─────────────────────────────────────────────────┤
│              NVIDIA GPU Hardware                 │
└─────────────────────────────────────────────────┘
```

---

## What Each Tool Does

| Tool | Stage | What it optimizes |
|---|---|---|
| **Megatron-LM / DeepSpeed** | Pre-training | Training from scratch on trillions of tokens |
| **Unsloth** | Fine-tuning | Backward pass — 2x faster, 70% less VRAM vs HuggingFace |
| **FlashInfer** | Inference (kernels) | Forward pass GPU math — attention, GEMM, MoE, sampling |
| **SGLang** | Inference (engine) | Serving — scheduling, batching, KV-cache (uses FlashInfer as primary backend) |
| **vLLM** | Inference (engine) | Serving — PagedAttention, continuous batching (uses FlashInfer optionally) |
| **TensorRT-LLM** | Inference (engine) | NVIDIA's engine — contributes kernels to FlashInfer |
| **NVIDIA Dynamo** | Orchestration | Multi-node routing, disaggregated prefill/decode, distributed KV-cache |

---

## Training vs Inference

```
┌─────────────────┐              ┌────────────────────┐
│    TRAINING      │              │    INFERENCE        │
│    (backward)    │  ── export   │    (forward only)   │
│                  │    weights   │                     │
│  Unsloth         │  ─────────►  │  SGLang / vLLM      │
│  (fine-tune)     │              │  FlashInfer kernels  │
└─────────────────┘              └────────────────────┘

 "Make the model                  "Serve the model
  smarter/specialized"             fast to users"
```

---

## FlashInfer Deep Dive

FlashInfer is the **shared kernel foundation** that multiple inference engines build on. Won **Best Paper at MLSys 2025**.

**Four kernel families:**
- **Attention** — paged KV-cache, prefill, decode, MLA (DeepSeek-style), cascade attention
- **GEMM** — matrix multiply with BF16, FP8, FP4 quantization
- **MoE** — fused Mixture of Experts routing and compute
- **Sampling** — top-k, top-p, speculative decoding

**Wraps multiple backends** and auto-selects the best for your hardware:
- FlashAttention-2/3, cuDNN, CUTLASS, TensorRT-LLM kernels

**Hardware support:** Turing (T4) → Ampere (A100) → Hopper (H100) → Blackwell (B200)

---

## Unsloth Deep Dive

| Feature | Details |
|---|---|
| **Purpose** | Fine-tune pre-trained LLMs (LoRA, QLoRA, full, RLHF/GRPO) |
| **Performance** | 2x faster training, 70% less VRAM vs standard HuggingFace |
| **Models** | 500+ — Llama, DeepSeek, Qwen, Gemma, GPT-OSS, vision, TTS |
| **Minimum** | Runs on Google Colab/Kaggle free tier, or locally with 3GB VRAM |

---

## NVIDIA Dynamo Deep Dive

Announced at GTC 2025. The orchestration layer for multi-node inference.

**Four key components:**
1. **Planner** — receives and plans request execution
2. **Smart Router** — decides which GPU/node handles each request
3. **Distributed KV Cache Manager** — coordinates cache across nodes
4. **NIXL** (NVIDIA Inference Transfer Library) — high-speed data transfer between nodes

**Key innovation:** disaggregated serving — separates prefill and decode phases onto different GPUs for independent optimization.

---

## Key Relationships

- **NVIDIA contributes kernels to FlashInfer** — strategy to get optimized code into the open-source ecosystem
- **SGLang + FlashInfer** — tightly coupled, co-developed (UC Berkeley), ~29% faster than vLLM on benchmarks
- **vLLM + FlashInfer** — optional backend, vLLM also has its own PagedAttention
- **Dynamo** doesn't use FlashInfer directly — it orchestrates engines that use FlashInfer internally
- **Unsloth** is complementary to all of the above — it produces the weights they serve
