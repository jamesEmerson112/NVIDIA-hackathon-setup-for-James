# Hackathon Quickstart — Inference Stack

*Created: 2026-03-15*

> For the hackathon, you're on the **inference/serving side only** — grab a pre-trained model, serve it, build your app on top. No training or fine-tuning during the event.
>
> Pairs with: [llm-inference-stack.md](./llm-inference-stack.md) | [github-references.md](./github-references.md) | [cloud-gpu-experiment-guide.md](./cloud-gpu-experiment-guide.md)

---

## Three Decisions

You need to make three independent choices:

```
  1. MODEL            2. ENGINE              3. ORCHESTRATION
  (the weights)       (serves the model)     (multi-GPU routing)

  Nemotron            SGLang  ──or──  vLLM   Dynamo (if multi-GPU)
  Llama                                      Skip (if single GPU)
  DeepSeek
  Qwen
```

---

## The Architecture (Corrected)

Dynamo **wraps** the engine — it's not a separate layer on top. FlashInfer is **inside** the engine, not a standalone layer you configure.

```
  ┌──────────────────────────────────────────────┐
  │  MODEL (the weights)                         │
  │  Nemotron, Llama, DeepSeek, Qwen, etc.       │
  └────────────────────┬─────────────────────────┘
                       │  loaded into
                       ▼
  ┌──────────────────────────────────────────────┐
  │              NVIDIA DYNAMO                   │
  │  (skip this layer for single-GPU)            │
  │                                              │
  │  Planner → Smart Router → NIXL transfers     │
  │  Distributed KV Cache Manager                │
  │                                              │
  │  ┌────────────────────────────────────────┐  │
  │  │  Backend (you pick one):              │  │
  │  │                                        │  │
  │  │   SGLang  ──or──  vLLM                 │  │
  │  │   (FlashInfer     (own kernels,        │  │
  │  │    built-in)       FlashInfer optional) │  │
  │  └──────────────────┬─────────────────────┘  │
  └─────────────────────┼───────────────────────┘
                        ▼
  ┌──────────────────────────────────────────────┐
  │             NVIDIA GPU HARDWARE              │
  │  T4 → A100 → H100 → B200                    │
  └──────────────────────────────────────────────┘
```

**Single GPU** skips Dynamo — you go straight from model into SGLang/vLLM.

You select the backend at launch:
```bash
# SGLang backend
python3 -m dynamo.sglang --model-path nvidia/nemotron-3-super-120b

# vLLM backend
python3 -m dynamo.vllm --model nvidia/nemotron-3-super-120b
```

---

## SGLang vs vLLM — Which Engine?

The choice depends on your app's **workload pattern**:

| | SGLang | vLLM |
|---|---|---|
| **KV cache strategy** | RadixAttention (cross-request prefix sharing) | PagedAttention (per-request memory efficiency) |
| **Shines when** | Many requests share common prefixes | Requests have diverse/unique context |
| **Best for** | Chatbots, agents, fixed system prompts, few-shot APIs | RAG with unique docs, batch processing diverse inputs |
| **Performance** | ~29% faster on benchmarks | More mature ecosystem |
| **FlashInfer** | Built-in (default backend) | Optional |
| **Dynamo support** | Disaggregated serving, KV-aware routing, multimodal | Same + LoRA, GB200 support |

**For most hackathon demos** (chatbots, agents, tool-use apps with shared system prompts) → **SGLang**.

**At hackathon scale** (low traffic, single GPU, one demo) the difference is negligible. Both work fine.

---

## No Local GPU? Options by Tier

### Tier 1: Zero GPU, Zero Cost (fastest setup)

**NVIDIA NIM API** — The recommended path for an NVIDIA hackathon.
- Sign up at [build.nvidia.com](https://build.nvidia.com) (free NVIDIA Developer Program)
- 1,000 free API credits on signup (up to 5,000)
- OpenAI-compatible API — no infrastructure at all
- Nemotron models available (Super 120B, Nano 30B, Nano 12B VL)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="nvapi-xxx"  # from build.nvidia.com
)

response = client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**OpenRouter** (backup) — Nemotron available at $0/million tokens. Use if NIM credits run out.

### Tier 2: Serverless GPU (serve your own model)

| Platform | Cost | Setup | Notes |
|---|---|---|---|
| **Modal + vLLM** | Free ($30/mo credits) | 15-30 min | Best DX, public URL, pure Python deploy |
| **RunPod Serverless** | ~$0.0004/sec | 30 min | vLLM template, pay only during inference |

### Tier 3: Rent a GPU (full control)

| Provider | GPU | Cost | Notes |
|---|---|---|---|
| **RunPod** | RTX 4090 – H100 | ~$0.34–1.19/hr | Best for hackathons — templates, Jupyter, SSH |
| **Lambda** | A100 – B200 | ~$1.48/hr | No egress fees, InfiniBand for multi-GPU |
| **Vast.ai** | Mixed marketplace | ~$0.50/hr | Cheapest, variable reliability |

### Don't use for serving
- **Google Colab / Kaggle** — Sessions disconnect. Fine for notebooks and fine-tuning, not for serving an endpoint during a demo.
- **Banana.dev** — Shut down (March 2024).

---

## NVIDIA Hackathon Bonus

NVIDIA hackathons typically provide:
- On-site hardware access (GB10 systems at GTC)
- Cloud credits via partners (e.g., $100 AWS credits for NIM on SageMaker)
- Free NIM API access

They consistently steer participants toward **NIM + Nemotron**. Using these is the path of least resistance and likely scores points with judges.

---

## Quickstart Checklist

1. [ ] Sign up at [build.nvidia.com](https://build.nvidia.com) — get NIM API key
2. [ ] Test a Nemotron API call (code example above)
3. [ ] Decide: API-only (Tier 1) or self-hosted (Tier 2/3)?
4. [ ] If self-hosted: pick SGLang or vLLM based on workload
5. [ ] Build your app on top of the OpenAI-compatible endpoint
6. [ ] Check if the hackathon provides credits/hardware (don't pay if you don't have to)

---

*Last updated: 2026-03-15*
