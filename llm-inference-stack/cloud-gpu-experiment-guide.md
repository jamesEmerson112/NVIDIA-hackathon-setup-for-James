# Cloud GPU Guide for LLM Experimentation

> No local GPU? No problem. This guide covers where to rent GPUs and how to run the inference stack hands-on.
>
> Pairs with: [llm-inference-stack.md](./llm-inference-stack.md) and [github-references.md](./github-references.md)

---

## Why Cloud GPUs

The inference stack we've studied (Unsloth → SGLang/vLLM → Dynamo) requires NVIDIA GPUs. Without one locally, cloud GPU rental is the practical path to hands-on experimentation.

**The cost mental model:**
- **Free tier** (Google Colab) — enough to fine-tune small models (1–3B params), zero cost
- **Pay-per-hour** (RunPod, Lambda, Vast.ai) — rent exactly the GPU you need, pay only while it's running
- **Self-hosted** — buy your own GPU (~$1,600+ for RTX 4090) — only worth it if you'll use it daily

For learning and experimentation, cloud is the right call.

---

## Provider Comparison

| Provider | GPU Options | Approx. Cost (A100 80GB) | Billing | Best For |
|---|---|---|---|---|
| **Google Colab (Free)** | T4 (15 GB VRAM) | Free | Session-based (max ~12h) | First experiments, models ≤3B |
| **Google Colab Pro** | T4, A100 (40 GB) | ~$10/mo | Session-based | Medium models (7–13B) |
| **RunPod** | RTX 4090, A100, H100, B200 | ~$1.19/hr | Per-second | Flexible experimentation |
| **Lambda** | A100, H100, GH200, B200 | ~$1.48/hr | Per-minute, no egress fees | Serious training, multi-GPU clusters |
| **Vast.ai** | Mixed marketplace (consumer + datacenter) | ~$0.50/hr | Per-second | Budget-conscious, tolerant of variability |
| **Thunder Compute** | A100 | ~$0.66/hr | Hourly | Simple workflow (VS Code extension) |

### Provider Pros & Cons

**Google Colab (Free)**
- ✅ Zero cost, instant setup, Unsloth has ready-made notebooks
- ❌ T4 only (15 GB VRAM), sessions disconnect, can't run SGLang server persistently

**RunPod**
- ✅ Per-second billing, great templates (Jupyter, SSH), wide GPU selection, good for Unsloth + SGLang workflow
- ❌ Popular GPUs can sell out during peak times

**Lambda**
- ✅ No egress fees, InfiniBand networking for multi-GPU, clean developer experience
- ❌ Frequent capacity shortages for popular GPUs, slightly higher prices than RunPod/Vast.ai

**Vast.ai**
- ✅ Cheapest option (50–70% less than hyperscalers), wide selection including consumer GPUs
- ❌ P2P marketplace — variable reliability, uptime, and security; machines can disappear mid-session

**Thunder Compute**
- ✅ VS Code extension for seamless dev, competitive A100 pricing
- ❌ Smaller provider, fewer GPU options

---

## GPU Selection Guide

How much VRAM do you need? It depends on model size and quantization:

| Model Size | Full Precision (FP16) | 4-bit Quantized (QLoRA) | Recommended GPU |
|---|---|---|---|
| **0.5–1B** | ~2–4 GB | ~1–2 GB | T4 (free Colab) |
| **3B** | ~6–12 GB | ~1.7–3 GB | T4 (free Colab) |
| **7–8B** | ~14–16 GB | ~4–6 GB | RTX 4090 (24 GB) or A100 40 GB |
| **13B** | ~26 GB | ~7–10 GB | A100 40 GB |
| **70B** | ~140 GB | ~35–40 GB | A100 80 GB or H100 80 GB |

**Rule of thumb:** 4-bit quantization (QLoRA) compresses model weights to ~25% of FP16 size, making it possible to fine-tune a 3B model on a free T4 or a 7B model on an RTX 4090.

---

## Recommended Experiment Path

Three progressive experiments, each building on the last. Ties directly to the pipeline from [llm-inference-stack.md](./llm-inference-stack.md):

```
Pre-trained model → [Unsloth fine-tune] → Export weights → [SGLang serve] → API
```

---

### Experiment 1: Free Tier — Colab + Unsloth

**Goal:** Fine-tune a small model, see how behavior changes before/after.

**Setup:**
- Google Colab (free tier, T4 GPU)
- Unsloth's ready-made Colab notebooks

**Steps:**
1. Open an Unsloth notebook from https://github.com/unslothai/notebooks
   - Recommended starting point: Llama 3.2 1B or Qwen 2.5 0.5B
2. Pick a small dataset (e.g., Alpaca-style instruction data, or your own Q&A pairs)
3. Run the notebook — fine-tunes with QLoRA (4-bit), takes ~30 min on free T4
4. Compare outputs: same prompt, base model vs fine-tuned model
5. Export the model (safetensors or GGUF format)

**What you learn:**
- How LoRA/QLoRA fine-tuning works in practice
- How small datasets change model behavior
- Unsloth's role in the stack (step 2 of the pipeline)

**Cost:** Free
**Time:** ~1–2 hours including setup

---

### Experiment 2: Paid Tier — RunPod + Unsloth + SGLang

**Goal:** Fine-tune a larger model AND serve it as an API — the full pipeline.

**Setup:**
- RunPod account (RTX 4090 pod, ~$0.34/hr)
- Jupyter Lab inside the pod

**Steps:**
1. Create a RunPod account and launch an RTX 4090 pod
2. Open Jupyter Lab from the pod dashboard
3. Install Unsloth:
   ```bash
   pip install unsloth
   ```
4. Load a 7B model (e.g., Llama 3.1 8B) with Unsloth — applies memory optimizations at import time
5. Attach LoRA adapters and fine-tune on your dataset
6. Export merged checkpoint (16-bit safetensors):
   ```python
   model.save_pretrained_merged("my-model", tokenizer)
   ```
7. Launch SGLang server from the pod terminal:
   ```bash
   python -m sglang.launch_server --model-path ./my-model --port 30002
   ```
8. Hit the API — it's OpenAI-compatible:
   ```python
   from openai import OpenAI
   client = OpenAI(base_url="http://localhost:30002/v1", api_key="none")
   response = client.chat.completions.create(
       model="my-model",
       messages=[{"role": "user", "content": "Hello!"}]
   )
   ```

**What you learn:**
- Full pipeline: fine-tune → export → serve → API call
- SGLang's role as the inference engine (step 4 of the pipeline)
- What it feels like to self-host an LLM endpoint

**Cost:** ~$1–3 for a 2–3 hour session on RTX 4090
**Time:** ~2–3 hours

> Reference walkthrough: [Fine-tuning and Deploying LLM with Unsloth, SGLang and RunPod](https://blog.dailydoseofds.com/p/fine-tuning-and-deploying-llm-with)

---

### Experiment 3: Multi-GPU (Optional) — Lambda or RunPod

**Goal:** Scale up — serve a larger model across multiple GPUs, optionally try Dynamo orchestration.

**Setup:**
- Lambda or RunPod multi-GPU instance (2–8× A100 or H100)

**Steps:**
1. Rent a multi-GPU instance (e.g., 2× A100 80 GB on Lambda, ~$2.96/hr)
2. Fine-tune a 70B model with QLoRA via Unsloth
3. Serve with SGLang using tensor parallelism:
   ```bash
   python -m sglang.launch_server --model-path ./my-70b-model --tp 2
   ```
4. (Advanced) Try NVIDIA Dynamo for disaggregated prefill/decode across GPUs

**What you learn:**
- How tensor parallelism splits a model across GPUs
- Dynamo's role as the orchestration layer (step 5 of the pipeline)
- When you actually need multi-GPU vs single-GPU

**Cost:** ~$6–12/hr depending on GPU choice
**Time:** Half a day

---

## Cost Estimates Summary

| Experiment | Provider | GPU | Duration | Estimated Cost |
|---|---|---|---|---|
| 1. First fine-tune | Google Colab | T4 (free) | 1–2 hrs | **Free** |
| 2. Full pipeline | RunPod | RTX 4090 | 2–3 hrs | **~$1–3** |
| 2. Full pipeline (alt) | Vast.ai | RTX 4090 | 2–3 hrs | **~$0.50–1.50** |
| 3. Multi-GPU | Lambda | 2× A100 80GB | 3–4 hrs | **~$9–12** |
| 3. Multi-GPU (alt) | RunPod | 2× A100 80GB | 3–4 hrs | **~$7–10** |

**Bottom line:** You can go from zero to serving your own fine-tuned LLM for under $5. The free Colab experiment costs nothing.

---

## Resources & Links

### Provider Signup
- Google Colab — https://colab.research.google.com
- RunPod — https://runpod.io
- Lambda — https://lambda.ai
- Vast.ai — https://vast.ai
- Thunder Compute — https://thundercompute.com

### Unsloth (Fine-Tuning)
- Unsloth GitHub — https://github.com/unslothai/unsloth
- Ready-made notebooks (250+) — https://github.com/unslothai/notebooks
- Colab quickstart — https://unsloth.ai/docs/get-started/install/google-colab
- SGLang deployment guide — https://unsloth.ai/docs/basics/inference-and-deployment/sglang-guide

### SGLang (Serving)
- SGLang GitHub — https://github.com/sgl-project/sglang

### Reference Articles
- [RunPod: Top Cloud GPU Providers](https://www.runpod.io/articles/guides/top-cloud-gpu-providers)
- [Lambda Pricing](https://lambda.ai/pricing)
- [Fine-tuning and Deploying LLM with Unsloth, SGLang and RunPod](https://blog.dailydoseofds.com/p/fine-tuning-and-deploying-llm-with)

---

*Last updated: 2026-03-15*
