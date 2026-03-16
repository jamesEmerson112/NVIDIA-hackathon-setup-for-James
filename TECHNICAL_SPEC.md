# Third Eye - Technical Feature Document (Comprehensive)

## Context

**Hackathon**: NVIDIA AI Agent Hackathon
**Problem**: Visually impaired people need real-time spatial awareness and navigation assistance.
**Solution**: "Third Eye" — a voice-only Mac desktop app that uses the webcam as a surrogate eye. Users speak to an AI agent, which captures and interprets the visual scene, provides surroundings descriptions, reads text/signs, and gives walking directions — all through voice.
**Team**: 2 people working in parallel. Person A handles Mac hardware (camera, mic, speakers). Person B handles NVIDIA AI model integrations (using test photos).

---

## Architecture Overview

```
 +======================================================================+
 |                        THIRD EYE (Python, macOS)                     |
 +======================================================================+
 |                                                                      |
 |  +------------------+        +-----------------------------------+   |
 |  |   MICROPHONE     |        |         SPEAKERS                  |   |
 |  |   (sounddevice)  |        |         (sounddevice)             |   |
 |  +--------+---------+        +----------------^------------------+   |
 |           |                                   |                      |
 |           v                                   |                      |
 |  +--------+---------+        +----------------+------------------+   |
 |  |   RIVA ASR       |        |         RIVA TTS                  |   |
 |  |   Speech-to-Text |        |         Text-to-Speech            |   |
 |  |   gRPC streaming |        |         gRPC batch                |   |
 |  +--------+---------+        +----------------^------------------+   |
 |           |                                   |                      |
 |           v                                   |                      |
 |  +--------+-------------------------------------------+----------+   |
 |  |                ORCHESTRATOR (orchestrator.py)                  |   |
 |  |                                                                |   |
 |  |  State Machine: LISTENING -> THINKING -> SPEAKING -> LISTENING |   |
 |  |                                                                |   |
 |  |  - Owns conversation_history: list[dict]                       |   |
 |  |  - Dispatches Nemotron tool_calls to handlers                  |   |
 |  |  - Manages continuous_mode background thread                   |   |
 |  |  - Mutes mic during TTS playback                               |   |
 |  +------+----------------+-------------------+-------------------+   |
 |         |                |                   |                       |
 |         v                v                   v                       |
 |  +------+------+  +-----+-------+  +--------+----------+            |
 |  |  NEMOTRON   |  |  WEBCAM +   |  |  NAVIGATION       |            |
 |  |  AGENT      |  |  LLAMA      |  |  (Apple MapKit     |            |
 |  |             |  |  VISION     |  |   via PyObjC)      |            |
 |  | OpenAI SDK  |  |             |  |                    |            |
 |  | Tool calling|  | cv2.Video   |  | CLGeocoder         |            |
 |  |             |  | Capture +   |  | MKDirections       |            |
 |  | Model:      |  | Llama 3.2   |  | Walking directions |            |
 |  | nemotron-3  |  | 90B Vision  |  |                    |            |
 |  | -super-120b |  | REST API    |  | Fallback: Google   |            |
 |  | -a12b       |  |             |  | Maps Directions    |            |
 |  +-------------+  +-------------+  +--------------------+            |
 |                                                                      |
 +======================================================================+

 NVIDIA Cloud APIs (all require NVIDIA_API_KEY env var):
   [1] Nemotron:     https://integrate.api.nvidia.com/v1/chat/completions
                     model: "nvidia/nemotron-3-super-120b-a12b"
   [2] Llama Vision: https://integrate.api.nvidia.com/v1/chat/completions
                     model: "meta/llama-3.2-90b-vision-instruct"
   [3] Riva ASR:     grpc.nvcf.nvidia.com:443  (gRPC + SSL)
   [4] Riva TTS:     grpc.nvcf.nvidia.com:443  (gRPC + SSL)
```

### Key Design Decisions
- **Synchronous + threading** (not asyncio) — simpler PyObjC/gRPC compatibility
- **Full conversation history** passed each Nemotron call — 1M context window makes this trivial for a demo
- **Mic muted during TTS playback** — prevents audio feedback loops
- **Nemotron as orchestrator** via tool calling — the LLM decides when to look, navigate, or read text
- **Nemotron is text-only** — all vision goes through Llama 3.2 Vision as a separate API call, results fed back to Nemotron as text

---

## Project Structure

```
third-eye/
  main.py                 # Entry point: CLI args, init all modules, run orchestrator
  orchestrator.py          # Central coordinator: state machine, event loop, tool dispatch
  agent.py                 # Nemotron client: OpenAI SDK, tool defs, agentic loop
  speech.py                # Riva ASR (streaming) + TTS (batch), with fallbacks
  vision.py                # cv2 webcam capture + Llama 3.2 Vision API client
  navigation.py            # Apple MapKit via PyObjC (or Google Maps fallback)
  config.py                # All API keys, endpoints, model IDs, constants
  prompts.py               # System prompt for Nemotron, VILA prompt templates
  audio_utils.py           # Low-level mic recording + speaker playback via sounddevice
  .env                     # NVIDIA_API_KEY=nvapi-xxx (gitignored)
  .env.example             # Template: NVIDIA_API_KEY=your-key-here
  requirements.txt         # All pip dependencies
  .gitignore               # Python + .env + __pycache__ + .DS_Store
  README.md                # Setup, deps, how to run
  demo/
    demo_script.md          # Scripted demo walkthrough for judges
  tests/
    test_agent.py
    test_vision.py
    test_speech.py
    test_navigation.py
  test_photos/              # Static test images for Person B to develop with
    street.jpg
    sign.jpg
    indoor.jpg
```

---

## Dependencies

### requirements.txt
```
# NVIDIA APIs
openai>=1.12.0                        # OpenAI-compatible SDK for Nemotron + Llama Vision
nvidia-riva-client>=2.14.0            # Riva ASR + TTS gRPC client
grpcio>=1.60.0                        # gRPC runtime (Riva dependency)
protobuf>=4.25.0                      # Protobuf (Riva dependency)

# Vision
opencv-python>=4.9.0                  # Webcam capture (cv2)

# Audio I/O
sounddevice>=0.4.6                    # Cross-platform mic input + speaker output
numpy>=1.26.0                         # Audio buffer manipulation

# macOS native (MapKit)
pyobjc-framework-MapKit>=10.1         # MapKit bindings
pyobjc-framework-CoreLocation>=10.1   # CLGeocoder, coordinates

# Configuration
python-dotenv>=1.0.0                  # Load .env file

# Fallbacks (install these too, in case Riva has issues)
SpeechRecognition>=3.10.0             # Fallback ASR (Google free tier)
pyttsx3>=2.90                         # Fallback TTS (macOS native say)
```

### System-level dependencies (macOS)
```bash
# Install these before pip install:
brew install portaudio    # Required by sounddevice
# Xcode Command Line Tools must be installed (for PyObjC compilation):
xcode-select --install
# Python 3.11+ recommended
```

---

## NVIDIA NIM & Tools Setup

This project uses **NVIDIA NIM APIs** for all AI model inference. NIM provides OpenAI-compatible endpoints — no GPU infrastructure required.

### Step 1: Get a NIM API Key

1. Sign up at [build.nvidia.com](https://build.nvidia.com) (free NVIDIA Developer Program)
2. You get **1,000 free API credits** on signup (up to 5,000)
3. Generate an API key — it starts with `nvapi-`
4. Add it to your `.env` file: `NVIDIA_API_KEY=nvapi-xxx`

### NIM Models Used in This Project

| Model | NIM Endpoint | Purpose |
|---|---|---|
| **Nemotron Super 120B** (`nvidia/nemotron-3-super-120b-a12b`) | `https://integrate.api.nvidia.com/v1/chat/completions` | Reasoning agent — orchestrates tools, synthesizes responses |
| **Llama 3.2 90B Vision** (`meta/llama-3.2-90b-vision-instruct`) | `https://integrate.api.nvidia.com/v1/chat/completions` | Scene analysis — describes webcam frames for the user |
| **Riva ASR** (Parakeet CTC 1.1B) | `grpc.nvcf.nvidia.com:443` (gRPC) | Speech-to-text — transcribes user voice input |
| **Riva TTS** (FastPitch HiFi-GAN) | `grpc.nvcf.nvidia.com:443` (gRPC) | Text-to-speech — speaks responses aloud |

All four models share the same `NVIDIA_API_KEY`. The chat completions endpoints are OpenAI-compatible (use the `openai` Python SDK). Riva uses gRPC with the `nvidia-riva-client` SDK.

### Quick API Test

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="nvapi-xxx"  # your key from build.nvidia.com
)

response = client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Other Available Nemotron Models (Smaller / Cheaper)

If you hit rate limits or want faster responses for testing:
- `nvidia/nemotron-3-nano-30b` — smaller reasoning model
- `nvidia/nemotron-3-nano-12b-vl` — small vision-language model (could replace Llama Vision)

### Backup: OpenRouter

If NIM credits run out, [OpenRouter](https://openrouter.ai) has Nemotron available at **$0/million tokens** (free tier). Same OpenAI SDK, different `base_url`:

```python
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-xxx"  # from openrouter.ai
)
```

### Hackathon Scoring Note

NVIDIA hackathons consistently steer participants toward **NIM + Nemotron**. Using these is the path of least resistance and likely scores points with judges. The project already uses both.

### Self-Hosted Option (If Needed)

If you need to self-host models instead of using NIM APIs (e.g., for latency, privacy, or if credits run out), the stack is:

```
Model Weights (Nemotron, Llama, etc.)
        │  loaded into
        ▼
Inference Engine (pick one):
  SGLang  ──or──  vLLM
        │
        ▼
NVIDIA GPU (rented or local)
```

**SGLang** is the recommended engine for this project (chatbot/agent with shared system prompts → RadixAttention prefix sharing helps). Launch with:
```bash
python -m sglang.launch_server --model-path nvidia/nemotron-3-super-120b --port 30002
```

Then point the `NEMOTRON_BASE_URL` in `config.py` to `http://localhost:30002/v1`.

**Cloud GPU options** (if no local GPU):

| Provider | GPU | Cost | Best For |
|---|---|---|---|
| **RunPod** | RTX 4090 – H100 | ~$0.34–1.19/hr | Hackathons — templates, Jupyter, SSH |
| **Lambda** | A100 – B200 | ~$1.48/hr | Multi-GPU, no egress fees |
| **Vast.ai** | Mixed marketplace | ~$0.50/hr | Cheapest, variable reliability |
| **Modal** | Serverless | Free $30/mo credits | Best DX, public URL, pure Python deploy |

For the hackathon, **stick with NIM APIs** (Tier 1 / zero infrastructure) unless you have a specific reason to self-host.

### NVIDIA Tools Reference

| Tool | What It Does | Relevant to This Project? |
|---|---|---|
| **NIM API** | Hosted inference (OpenAI-compatible) | **Yes** — all our model calls go here |
| **Riva** | Speech AI (ASR + TTS) | **Yes** — voice I/O |
| **SGLang** | Self-hosted inference engine | Only if self-hosting |
| **vLLM** | Self-hosted inference engine (alternative) | Only if self-hosting |
| **NVIDIA Dynamo** | Multi-GPU orchestration | Only for multi-node self-hosting |
| **FlashInfer** | GPU kernel library (inside SGLang/vLLM) | Internal to engines, not configured directly |
| **Unsloth** | Fine-tuning (LoRA/QLoRA) | Not needed — using pre-trained models |
| **NeMo** | Model training & deployment framework | Not needed for this project |
| **TensorRT-LLM** | NVIDIA's compiled inference engine | Alternative to SGLang/vLLM if self-hosting |

### GitHub Repos

- **SGLang** — https://github.com/sgl-project/sglang
- **vLLM** — https://github.com/vllm-project/vllm
- **TensorRT-LLM** — https://github.com/NVIDIA/TensorRT-LLM
- **NVIDIA Dynamo** — https://github.com/ai-dynamo/dynamo
- **FlashInfer** — https://github.com/flashinfer-ai/flashinfer
- **Unsloth** — https://github.com/unslothai/unsloth
- **NVIDIA Nemotron** — https://github.com/NVIDIA-NeMo/Nemotron
- **NVIDIA NeMo** — https://github.com/NVIDIA-NeMo

---

## File-by-File Implementation Specification

---

### `config.py` — Configuration

```python
"""
All configuration constants for Third Eye.
Loads NVIDIA_API_KEY from .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not set. Create a .env file with NVIDIA_API_KEY=nvapi-xxx")

# --- Nemotron (Reasoning Agent) ---
NEMOTRON_BASE_URL = "https://integrate.api.nvidia.com/v1"
NEMOTRON_MODEL = "nvidia/nemotron-3-super-120b-a12b"
NEMOTRON_MAX_TOKENS = 2048
NEMOTRON_TEMPERATURE = 0.6

# --- Llama 3.2 Vision (Scene Analysis) ---
VISION_BASE_URL = "https://integrate.api.nvidia.com/v1"
VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"
VISION_MAX_TOKENS = 512
VISION_TEMPERATURE = 0.2

# --- Riva ASR (Speech-to-Text) ---
RIVA_URI = "grpc.nvcf.nvidia.com:443"
RIVA_ASR_FUNCTION_ID = "1598d209-5e27-4d3c-8079-4751568b1081"  # Parakeet CTC 1.1B
RIVA_ASR_SAMPLE_RATE = 16000
RIVA_ASR_LANGUAGE = "en-US"

# --- Riva TTS (Text-to-Speech) ---
RIVA_TTS_FUNCTION_ID = "0149dedb-2be8-4195-b9a0-e57e0e14f972"  # FastPitch HiFi-GAN
RIVA_TTS_VOICE = "English-US.Female-1"
RIVA_TTS_SAMPLE_RATE = 44100
RIVA_TTS_LANGUAGE = "en-US"

# --- Webcam ---
WEBCAM_DEVICE_INDEX = 0       # 0 = default built-in camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 60             # Lower = smaller payload, faster upload

# --- Continuous Mode ---
CONTINUOUS_MODE_INTERVAL = 5  # seconds between frame captures in continuous mode

# --- Fallback flags ---
USE_RIVA_ASR = True           # Set False to use SpeechRecognition fallback
USE_RIVA_TTS = True           # Set False to use pyttsx3 fallback
```

---

### `prompts.py` — System Prompt and Vision Prompt Templates

```python
"""
All prompts used by the system.
"""

# System prompt sent to Nemotron as the first message in every conversation
SYSTEM_PROMPT = """You are Third Eye, an AI navigation and surroundings assistant \
for visually impaired people. You communicate through voice only — the user speaks \
to you and you speak back.

YOUR CAPABILITIES:
- Capture and describe what the user's webcam sees (their surroundings)
- Provide walking directions to destinations
- Read text from signs, labels, and documents in view
- Continuous monitoring mode that proactively alerts about changes

INTERACTION GUIDELINES:
- Keep responses concise and clear — they will be read aloud via text-to-speech
- Use spatial language: "on your left", "directly ahead", "about 10 feet away"
- Prioritize safety-critical information: obstacles, vehicles, stairs, curbs
- Be warm but efficient — avoid filler words and unnecessary pleasantries
- When describing surroundings, organize by relevance: immediate dangers first, \
then navigation-relevant features, then general context
- Use approximate distances and clock positions when helpful ("at your 2 o'clock")
- If you need to use a tool, do so without announcing it — just provide the result naturally
- Never output markdown, bullet points, or formatted text — speak naturally as if talking to someone
- Never say "I'm going to use a tool" or "Let me call a function" — just do it silently

CONTEXT:
- The user is visually impaired and relying on you for spatial awareness
- You are running on their Mac laptop with a webcam
- Location is provided verbally by the user (no GPS)
- You remember the full conversation and recent scene descriptions
- When the user asks about their surroundings, ALWAYS use the capture_and_describe tool
- When the user asks to go somewhere, ALWAYS use the get_directions tool
- When the user asks to read something, ALWAYS use the read_text tool
"""

# Prompt sent to Llama Vision for general scene description
VISION_DESCRIBE_PROMPT = """Describe what you see in this image for a visually impaired person \
who is navigating a physical space. Focus on:
1. IMMEDIATE SAFETY: obstacles, vehicles, stairs, curbs, uneven surfaces, construction
2. SPATIAL LAYOUT: what is directly ahead, to the left, to the right, on the ground
3. NAVIGATION AIDS: doors, crosswalks, traffic signals, ramps, elevators, signs
4. PEOPLE AND ACTIVITY: people nearby, their approximate distance and direction
5. GENERAL CONTEXT: indoor/outdoor, lighting conditions, weather if visible

Use spatial language like "directly ahead", "to your left", "about 10 feet away". \
Be concise but thorough. Prioritize safety-critical information."""

# Prompt sent to Llama Vision when focus is specified (e.g., "obstacles", "people")
VISION_FOCUSED_PROMPT_TEMPLATE = """Describe what you see in this image for a visually impaired \
person, focusing specifically on: {focus}

Use spatial language like "directly ahead", "to your left", "about 10 feet away". \
Be concise and specific to the requested focus area."""

# Prompt sent to Llama Vision for text/sign reading
VISION_READ_TEXT_PROMPT = """Read ALL text visible in this image. This includes:
- Street signs, building names, store signs
- Notices, menus, labels
- Screen text, documents
- Numbers, addresses
- Warning signs, safety notices

For each piece of text found, describe where it is in the image \
(e.g., "on the sign directly ahead", "on the door to the left"). \
If no text is visible, say so clearly."""

# Prompt sent to Nemotron during continuous mode updates
CONTINUOUS_MODE_PROMPT_TEMPLATE = """CONTINUOUS NAVIGATION UPDATE:

Previous scene description (from {seconds_ago} seconds ago):
{previous_description}

Current scene description (just captured):
{current_description}

Current navigation step (if navigating): {current_nav_step}

Based on the changes between the previous and current scene, provide a brief \
alert to the user ONLY if there is something important they should know about. \
Important changes include: new obstacles, approaching vehicles, upcoming turns, \
changes in terrain, arriving at destination.

If nothing significant has changed, respond with exactly: NO_UPDATE

Keep any alert to 1-2 sentences maximum."""
```

---

### `agent.py` — Nemotron Agent with Tool Calling

```python
"""
Nemotron agent client with tool calling support.
Uses OpenAI SDK pointed at NVIDIA's API endpoint.
Implements the agentic loop: call LLM -> detect tool_calls -> dispatch -> feed back -> repeat.
"""
import json
from openai import OpenAI
from config import (
    NVIDIA_API_KEY, NEMOTRON_BASE_URL, NEMOTRON_MODEL,
    NEMOTRON_MAX_TOKENS, NEMOTRON_TEMPERATURE
)
from prompts import SYSTEM_PROMPT

# --- OpenAI-compatible client pointed at NVIDIA ---
client = OpenAI(
    base_url=NEMOTRON_BASE_URL,
    api_key=NVIDIA_API_KEY
)

# --- Tool Definitions (JSON schemas for Nemotron's function calling) ---
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "capture_and_describe",
            "description": (
                "Captures a photo from the user's webcam and analyzes it using "
                "a vision AI model. Returns a detailed text description of the "
                "user's current surroundings including obstacles, people, vehicles, "
                "signs, terrain, crosswalks, doors, and anything else relevant to "
                "a visually impaired person navigating the physical world. Use this "
                "tool whenever the user asks about what is around them, in front of "
                "them, or needs visual information about their environment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": (
                            "Optional focus area for the description. Examples: "
                            "'obstacles', 'text and signs', 'people', 'traffic'. "
                            "If not provided, gives a general surroundings description."
                        )
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_directions",
            "description": (
                "Gets step-by-step walking directions from the user's current "
                "location to a specified destination. Returns a list of navigation "
                "steps with distances. Use this when the user asks to go somewhere, "
                "requests directions, or mentions a destination."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "The destination address or place name."
                    },
                    "origin": {
                        "type": "string",
                        "description": (
                            "The starting location. If not provided, uses the "
                            "user's last stated location."
                        )
                    }
                },
                "required": ["destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_continuous_mode",
            "description": (
                "Toggles continuous navigation mode on or off. When enabled, the "
                "system periodically captures webcam frames and proactively alerts "
                "the user about important changes in their surroundings. Use this "
                "when the user asks to start or stop continuous guidance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "True to enable, false to disable."
                    },
                    "interval_seconds": {
                        "type": "number",
                        "description": "Seconds between captures. Default 5."
                    }
                },
                "required": ["enabled"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_text",
            "description": (
                "Captures a photo and reads all visible text — signs, labels, "
                "menus, documents, screens, etc. Use when the user asks 'what "
                "does that sign say', 'read that for me', 'is there any text'."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def call_nemotron(conversation_history: list[dict]) -> dict:
    """
    Send conversation history to Nemotron and get a response.
    Returns the full response choice (may contain tool_calls or text content).

    Args:
        conversation_history: List of message dicts with 'role' and 'content' keys.
                              Must start with system message.

    Returns:
        The response choice object from the API.
    """
    response = client.chat.completions.create(
        model=NEMOTRON_MODEL,
        messages=conversation_history,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto",
        temperature=NEMOTRON_TEMPERATURE,
        max_tokens=NEMOTRON_MAX_TOKENS,
    )
    return response.choices[0]


def run_agentic_loop(conversation_history: list[dict], tool_handlers: dict) -> str:
    """
    Run the agentic loop: call Nemotron, handle tool calls, repeat until text response.

    This is the core agent logic. Nemotron may respond with tool_calls, in which case
    we execute each tool, feed the results back, and call Nemotron again. This repeats
    until Nemotron produces a final text response (no more tool_calls).

    Args:
        conversation_history: The full conversation (mutated in place — tool call
                              messages and tool result messages are appended).
        tool_handlers: Dict mapping tool names to callable functions.
                       Each function receives **kwargs matching the tool's parameters
                       and returns a string result.
                       Example: {"capture_and_describe": vision.capture_and_describe}

    Returns:
        The final text response from Nemotron (to be spoken via TTS).
    """
    max_iterations = 5  # Safety limit to prevent infinite tool-calling loops

    for _ in range(max_iterations):
        choice = call_nemotron(conversation_history)

        # If Nemotron produced a text response (no tool calls), we're done
        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
            assistant_message = choice.message.content or ""
            conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            return assistant_message

        # Nemotron wants to call tools — append the assistant message with tool_calls
        conversation_history.append({
            "role": "assistant",
            "content": choice.message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in choice.message.tool_calls
            ]
        })

        # Execute each tool call and append results
        for tool_call in choice.message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

            # Dispatch to the appropriate handler
            if func_name in tool_handlers:
                try:
                    result = tool_handlers[func_name](**func_args)
                except Exception as e:
                    result = f"Error executing {func_name}: {str(e)}"
            else:
                result = f"Unknown tool: {func_name}"

            # Append tool result message
            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

    # If we hit the iteration limit, return whatever we have
    return "I'm having trouble processing that request. Could you try again?"


def create_initial_history() -> list[dict]:
    """Create a fresh conversation history with the system prompt."""
    return [{"role": "system", "content": SYSTEM_PROMPT}]
```

---

### `vision.py` — Webcam Capture + Llama 3.2 Vision API

```python
"""
Handles webcam frame capture via OpenCV and scene analysis via Llama 3.2 Vision API.
Two main functions:
  - capture_frame(): grabs a JPEG frame from the webcam
  - analyze_frame(): sends a frame to Llama Vision and returns text description
  - capture_and_describe(): combines both (used as tool handler)
  - read_text(): captures and reads visible text (used as tool handler)
"""
import cv2
import base64
from openai import OpenAI
from config import (
    NVIDIA_API_KEY, VISION_BASE_URL, VISION_MODEL,
    VISION_MAX_TOKENS, VISION_TEMPERATURE,
    WEBCAM_DEVICE_INDEX, FRAME_WIDTH, FRAME_HEIGHT, JPEG_QUALITY
)
from prompts import (
    VISION_DESCRIBE_PROMPT, VISION_FOCUSED_PROMPT_TEMPLATE,
    VISION_READ_TEXT_PROMPT
)

# --- Vision API client (same OpenAI SDK, different model) ---
vision_client = OpenAI(
    base_url=VISION_BASE_URL,
    api_key=NVIDIA_API_KEY
)

# --- Webcam handle (kept open for low-latency repeated captures) ---
_camera = None


def _get_camera():
    """Lazy-initialize and return the webcam handle."""
    global _camera
    if _camera is None or not _camera.isOpened():
        _camera = cv2.VideoCapture(WEBCAM_DEVICE_INDEX)
        _camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not _camera.isOpened():
            raise RuntimeError("Could not open webcam. Check camera permissions in System Settings > Privacy > Camera.")
    return _camera


def capture_frame() -> bytes:
    """
    Capture a single frame from the webcam and return it as JPEG bytes.

    Returns:
        JPEG-encoded image bytes, resized to FRAME_WIDTH x FRAME_HEIGHT.

    Raises:
        RuntimeError: If webcam cannot be opened or frame cannot be captured.
    """
    camera = _get_camera()
    ret, frame = camera.read()
    if not ret:
        raise RuntimeError("Failed to capture frame from webcam.")

    # Resize to target dimensions
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Encode as JPEG with specified quality
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    _, jpeg_bytes = cv2.imencode('.jpg', frame, encode_params)
    return jpeg_bytes.tobytes()


def analyze_frame(jpeg_bytes: bytes, prompt: str) -> str:
    """
    Send a JPEG frame to Llama 3.2 Vision API and get a text description.

    The image is base64-encoded and sent as part of the message content
    using the OpenAI vision message format.

    Args:
        jpeg_bytes: JPEG-encoded image bytes.
        prompt: The text prompt to send alongside the image.

    Returns:
        Text description from the vision model.
    """
    b64_image = base64.b64encode(jpeg_bytes).decode("utf-8")

    response = vision_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=VISION_MAX_TOKENS,
        temperature=VISION_TEMPERATURE,
    )
    return response.choices[0].message.content


def capture_and_describe(focus: str = None) -> str:
    """
    Tool handler: capture a webcam frame and describe the scene.
    This is called by the Nemotron agentic loop when it invokes the
    'capture_and_describe' tool.

    Args:
        focus: Optional focus area (e.g., "obstacles", "people", "traffic").
               If None, uses the general description prompt.

    Returns:
        Text description of the scene from Llama Vision.
    """
    frame = capture_frame()
    if focus:
        prompt = VISION_FOCUSED_PROMPT_TEMPLATE.format(focus=focus)
    else:
        prompt = VISION_DESCRIBE_PROMPT
    return analyze_frame(frame, prompt)


def read_text() -> str:
    """
    Tool handler: capture a webcam frame and read all visible text.
    Called by the Nemotron agentic loop for the 'read_text' tool.

    Returns:
        All text detected in the scene, with location descriptions.
    """
    frame = capture_frame()
    return analyze_frame(frame, VISION_READ_TEXT_PROMPT)


def release_camera():
    """Release the webcam handle. Call on shutdown."""
    global _camera
    if _camera is not None:
        _camera.release()
        _camera = None
```

**For Person B testing with static photos instead of webcam:**

```python
# In tests or during development, replace capture_frame() with:
def capture_frame_from_file(path: str) -> bytes:
    """Load a test photo as JPEG bytes (for development without webcam)."""
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"Could not read test image: {path}")
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    _, jpeg_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return jpeg_bytes.tobytes()
```

---

### `speech.py` — Riva ASR + TTS (with Fallbacks)

```python
"""
Voice I/O module wrapping NVIDIA Riva ASR (speech-to-text) and TTS (text-to-speech).
Includes fallbacks to SpeechRecognition + pyttsx3 if Riva is unavailable.

Key functions:
  - listen() -> str: Record from mic, return transcript
  - speak(text) -> None: Convert text to audio and play
"""
import numpy as np
import sounddevice as sd
from config import (
    NVIDIA_API_KEY, RIVA_URI,
    RIVA_ASR_FUNCTION_ID, RIVA_ASR_SAMPLE_RATE, RIVA_ASR_LANGUAGE,
    RIVA_TTS_FUNCTION_ID, RIVA_TTS_VOICE, RIVA_TTS_SAMPLE_RATE, RIVA_TTS_LANGUAGE,
    USE_RIVA_ASR, USE_RIVA_TTS
)

# --- Riva clients (initialized lazily) ---
_asr_service = None
_tts_service = None


def _init_riva_asr():
    """Initialize Riva ASR gRPC client."""
    global _asr_service
    import riva.client
    auth = riva.client.Auth(
        ssl_cert=None,
        use_ssl=True,
        uri=RIVA_URI,
        metadata_args=[
            ("function-id", RIVA_ASR_FUNCTION_ID),
            ("authorization", f"Bearer {NVIDIA_API_KEY}"),
        ]
    )
    _asr_service = riva.client.ASRService(auth)
    return _asr_service


def _init_riva_tts():
    """Initialize Riva TTS gRPC client."""
    global _tts_service
    import riva.client
    auth = riva.client.Auth(
        ssl_cert=None,
        use_ssl=True,
        uri=RIVA_URI,
        metadata_args=[
            ("function-id", RIVA_TTS_FUNCTION_ID),
            ("authorization", f"Bearer {NVIDIA_API_KEY}"),
        ]
    )
    _tts_service = riva.client.SpeechSynthesisService(auth)
    return _tts_service


def listen() -> str:
    """
    Record audio from the microphone and return the transcript.

    Uses Riva ASR if USE_RIVA_ASR is True, otherwise falls back to
    SpeechRecognition with Google's free API.

    Implementation approach for MVP:
    - Press Enter to start recording
    - Record for up to 10 seconds or until 2 seconds of silence
    - Send to ASR and return transcript

    Returns:
        Transcribed text from the user's speech.
    """
    if USE_RIVA_ASR:
        return _listen_riva()
    else:
        return _listen_fallback()


def _listen_riva() -> str:
    """Record audio and transcribe using Riva ASR streaming."""
    import riva.client

    if _asr_service is None:
        _init_riva_asr()

    # Record audio from microphone
    duration_seconds = 10  # max recording duration
    print("Listening... (speak now)")
    audio_data = sd.rec(
        int(duration_seconds * RIVA_ASR_SAMPLE_RATE),
        samplerate=RIVA_ASR_SAMPLE_RATE,
        channels=1,
        dtype='int16'
    )

    # Wait for user to press Enter to stop, or record for full duration
    # For MVP: simple fixed-duration or silence-detection approach
    input("Press Enter when done speaking...")
    sd.stop()

    # Trim silence from end (basic approach)
    audio_bytes = audio_data.tobytes()

    # Use offline (batch) recognition for simplicity
    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        language_code=RIVA_ASR_LANGUAGE,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        sample_rate_hertz=RIVA_ASR_SAMPLE_RATE,
        audio_channel_count=1,
    )

    response = _asr_service.offline_recognize(audio_bytes, config)

    if response.results:
        transcript = response.results[0].alternatives[0].transcript
        print(f"You said: {transcript}")
        return transcript
    return ""


def _listen_fallback() -> str:
    """Fallback: use SpeechRecognition with Google's free API."""
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:
        print("Listening... (speak now)")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)

    try:
        transcript = recognizer.recognize_google(audio)
        print(f"You said: {transcript}")
        return transcript
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"ASR error: {e}")
        return ""


def speak(text: str) -> None:
    """
    Convert text to speech and play through speakers.

    Uses Riva TTS if USE_RIVA_TTS is True, otherwise falls back to pyttsx3.
    This function blocks until audio playback is complete.

    Args:
        text: The text to speak aloud.
    """
    if not text or text.strip() == "":
        return

    if USE_RIVA_TTS:
        _speak_riva(text)
    else:
        _speak_fallback(text)


def _speak_riva(text: str) -> None:
    """Synthesize and play speech using Riva TTS."""
    import riva.client

    if _tts_service is None:
        _init_riva_tts()

    response = _tts_service.synthesize(
        text=text,
        voice_name=RIVA_TTS_VOICE,
        language_code=RIVA_TTS_LANGUAGE,
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hz=RIVA_TTS_SAMPLE_RATE,
    )

    # Convert raw PCM bytes to numpy array and play
    audio_array = np.frombuffer(response.audio, dtype=np.int16)
    sd.play(audio_array, samplerate=RIVA_TTS_SAMPLE_RATE)
    sd.wait()  # Block until playback finishes


def _speak_fallback(text: str) -> None:
    """Fallback: use pyttsx3 (macOS native TTS)."""
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)  # Slightly slower for clarity
    engine.say(text)
    engine.runAndWait()
```

---

### `navigation.py` — Apple MapKit via PyObjC (with Fallback)

```python
"""
Navigation module: gets walking directions using Apple MapKit via PyObjC.
Falls back to a simplified text-based response if MapKit is unavailable.

Key function:
  - get_directions(destination, origin=None) -> str
"""
import threading
from config import NVIDIA_API_KEY


def get_directions(destination: str, origin: str = None) -> str:
    """
    Tool handler: get walking directions from origin to destination.

    Uses Apple MapKit via PyObjC for real directions.
    Falls back to a text description if MapKit fails.

    Args:
        destination: Place name or address (e.g., "Starbucks on Market St")
        origin: Starting location (optional, e.g., "500 Howard St, San Francisco")

    Returns:
        Formatted string with step-by-step walking directions.
    """
    try:
        return _get_directions_mapkit(destination, origin)
    except Exception as e:
        print(f"MapKit failed ({e}), using fallback")
        return _get_directions_fallback(destination, origin)


def _get_directions_mapkit(destination: str, origin: str = None) -> str:
    """Get walking directions using Apple MapKit via PyObjC."""
    import MapKit
    import CoreLocation
    from PyObjCTools import AppHelper

    result = {"steps": None, "error": None}
    event = threading.Event()

    def _do_geocode_and_directions():
        geocoder = CoreLocation.CLGeocoder.alloc().init()

        def on_dest_geocoded(placemarks, error):
            if error or not placemarks:
                result["error"] = f"Could not find location: {destination}"
                event.set()
                return

            dest_placemark = placemarks[0]
            dest_mapitem = MapKit.MKMapItem.alloc().initWithPlacemark_(
                MapKit.MKPlacemark.alloc().initWithPlacemark_(dest_placemark)
            )

            # If origin provided, geocode it too; otherwise use a default
            if origin:
                def on_origin_geocoded(origin_pms, origin_err):
                    if origin_err or not origin_pms:
                        result["error"] = f"Could not find origin: {origin}"
                        event.set()
                        return
                    origin_placemark = origin_pms[0]
                    origin_mapitem = MapKit.MKMapItem.alloc().initWithPlacemark_(
                        MapKit.MKPlacemark.alloc().initWithPlacemark_(origin_placemark)
                    )
                    _calculate_directions(origin_mapitem, dest_mapitem)

                geocoder.geocodeAddressString_completionHandler_(origin, on_origin_geocoded)
            else:
                # Use current location (requires Location Services permission)
                origin_mapitem = MapKit.MKMapItem.mapItemForCurrentLocation()
                _calculate_directions(origin_mapitem, dest_mapitem)

        def _calculate_directions(source_item, dest_item):
            request = MapKit.MKDirections.Request.alloc().init()
            request.setSource_(source_item)
            request.setDestination_(dest_item)
            request.setTransportType_(1)  # MKDirectionsTransportTypeWalking = 1

            directions = MapKit.MKDirections.alloc().initWithRequest_(request)

            def on_directions_calculated(response, error):
                if error:
                    result["error"] = f"Could not calculate directions: {error}"
                    event.set()
                    return

                route = response.routes()[0]
                steps = []
                for step in route.steps():
                    instruction = step.instructions()
                    distance = step.distance()
                    if instruction:  # Skip empty steps
                        if distance > 0:
                            # Convert meters to feet for US
                            feet = int(distance * 3.281)
                            steps.append(f"{instruction} for {feet} feet")
                        else:
                            steps.append(instruction)

                total_distance = int(route.distance() * 3.281)
                total_time = int(route.expectedTravelTime() / 60)

                result["steps"] = {
                    "steps": steps,
                    "total_distance_feet": total_distance,
                    "total_time_minutes": total_time,
                }
                event.set()

            directions.calculateDirectionsWithCompletionHandler_(on_directions_calculated)

        geocoder.geocodeAddressString_completionHandler_(destination, on_dest_geocoded)

    # Run on main thread (required by MapKit)
    _do_geocode_and_directions()

    # Wait for async completion (timeout after 15 seconds)
    event.wait(timeout=15)

    if result["error"]:
        return f"Navigation error: {result['error']}"

    if result["steps"] is None:
        return "Could not get directions. The request timed out."

    # Format as readable text
    steps_data = result["steps"]
    lines = [f"Walking directions to {destination}:"]
    lines.append(f"Total distance: about {steps_data['total_distance_feet']} feet")
    lines.append(f"Estimated time: about {steps_data['total_time_minutes']} minutes")
    lines.append("")
    for i, step in enumerate(steps_data["steps"], 1):
        lines.append(f"Step {i}: {step}")

    return "\n".join(lines)


def _get_directions_fallback(destination: str, origin: str = None) -> str:
    """
    Fallback when MapKit is unavailable.
    Returns a message asking the user to provide more context,
    since we can't actually calculate directions without a maps API.

    In a real fallback, you would use Google Maps Directions API here:
    https://maps.googleapis.com/maps/api/directions/json?origin=...&destination=...&mode=walking&key=...
    """
    origin_text = f" from {origin}" if origin else ""
    return (
        f"I'd like to help you get to {destination}{origin_text}, but I'm having "
        f"trouble accessing the maps service right now. Could you ask someone nearby "
        f"for directions, or try describing landmarks around you so I can help orient you?"
    )
```

---

### `audio_utils.py` — Low-Level Audio I/O

```python
"""
Low-level audio I/O utilities using sounddevice.
Handles microphone recording and speaker playback.
"""
import numpy as np
import sounddevice as sd


def record_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """
    Record audio from the default microphone.

    Args:
        duration: Recording duration in seconds.
        sample_rate: Sample rate in Hz (default 16000 for speech).

    Returns:
        NumPy array of int16 audio samples.
    """
    print(f"Recording for {duration}s...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    return audio.flatten()


def play_audio(audio_data: np.ndarray, sample_rate: int = 44100) -> None:
    """
    Play audio through the default speakers. Blocks until complete.

    Args:
        audio_data: NumPy array of audio samples.
        sample_rate: Sample rate in Hz.
    """
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()


def get_audio_devices():
    """List available audio input/output devices (for debugging)."""
    return sd.query_devices()
```

---

### `orchestrator.py` — Central Coordinator

```python
"""
Central orchestrator: manages the state machine, conversation history,
tool dispatch, and continuous mode.

State Machine:
  LISTENING -> THINKING -> SPEAKING -> LISTENING

The orchestrator mutes the microphone during SPEAKING to prevent
feedback loops. In continuous mode, it runs a background thread that
periodically captures and analyzes frames.
"""
import threading
import time
from enum import Enum
from agent import run_agentic_loop, create_initial_history
from speech import listen, speak
from vision import capture_and_describe, read_text, release_camera
from navigation import get_directions
from config import CONTINUOUS_MODE_INTERVAL
from prompts import CONTINUOUS_MODE_PROMPT_TEMPLATE


class AppState(Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class Orchestrator:
    def __init__(self):
        self.state = AppState.LISTENING
        self.conversation_history = create_initial_history()
        self.continuous_mode = False
        self.continuous_timer = None
        self.continuous_interval = CONTINUOUS_MODE_INTERVAL
        self.last_scene_description = None
        self.last_scene_time = None
        self.current_nav_step = "No active navigation"
        self.running = True

        # Tool handlers map: tool_name -> callable
        # These are the functions that get called when Nemotron invokes a tool
        self.tool_handlers = {
            "capture_and_describe": capture_and_describe,
            "read_text": read_text,
            "get_directions": get_directions,
            "toggle_continuous_mode": self._handle_toggle_continuous,
        }

    def _handle_toggle_continuous(self, enabled: bool, interval_seconds: float = None) -> str:
        """Handle the toggle_continuous_mode tool call."""
        if interval_seconds:
            self.continuous_interval = interval_seconds

        if enabled and not self.continuous_mode:
            self.continuous_mode = True
            self._start_continuous_timer()
            return f"Continuous mode enabled. I'll check your surroundings every {self.continuous_interval} seconds."
        elif not enabled and self.continuous_mode:
            self.continuous_mode = False
            self._stop_continuous_timer()
            return "Continuous mode disabled."
        elif enabled:
            return "Continuous mode is already on."
        else:
            return "Continuous mode is already off."

    def _start_continuous_timer(self):
        """Start the background timer for continuous frame capture."""
        if self.continuous_timer:
            self.continuous_timer.cancel()

        def _continuous_loop():
            while self.continuous_mode and self.running:
                # Only capture if we're not currently processing a user request
                if self.state == AppState.LISTENING:
                    self._process_continuous_update()
                time.sleep(self.continuous_interval)

        self.continuous_timer = threading.Thread(target=_continuous_loop, daemon=True)
        self.continuous_timer.start()

    def _stop_continuous_timer(self):
        """Stop the continuous mode timer."""
        self.continuous_mode = False
        # Thread will exit on its own since self.continuous_mode is False

    def _process_continuous_update(self):
        """Capture a frame, analyze it, and alert user if something changed."""
        try:
            current_description = capture_and_describe()

            if self.last_scene_description:
                seconds_ago = int(time.time() - self.last_scene_time) if self.last_scene_time else self.continuous_interval

                # Ask Nemotron to compare and decide if an alert is needed
                update_prompt = CONTINUOUS_MODE_PROMPT_TEMPLATE.format(
                    seconds_ago=seconds_ago,
                    previous_description=self.last_scene_description,
                    current_description=current_description,
                    current_nav_step=self.current_nav_step,
                )

                # Use a temporary conversation for the continuous update
                # (don't pollute the main conversation history with every frame)
                temp_history = self.conversation_history.copy()
                temp_history.append({"role": "user", "content": update_prompt})

                response = run_agentic_loop(temp_history, self.tool_handlers)

                if response.strip() != "NO_UPDATE":
                    # There's something to tell the user
                    self.state = AppState.SPEAKING
                    speak(response)
                    self.state = AppState.LISTENING
                    # Add the alert to main conversation history for context
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": f"[Continuous mode alert] {response}"
                    })

            self.last_scene_description = current_description
            self.last_scene_time = time.time()

        except Exception as e:
            print(f"Continuous mode error: {e}")

    def run(self):
        """
        Main event loop. Runs until self.running is set to False.

        Loop:
          1. LISTENING: Wait for user speech via ASR
          2. THINKING: Send to Nemotron, handle tool calls
          3. SPEAKING: Play response via TTS
          4. Back to LISTENING
        """
        speak("Third Eye is ready. How can I help you?")

        while self.running:
            try:
                # --- LISTENING ---
                self.state = AppState.LISTENING
                user_input = listen()

                if not user_input:
                    continue  # No speech detected, loop back

                # Check for exit commands
                if user_input.lower().strip() in ["exit", "quit", "stop", "goodbye", "bye"]:
                    speak("Goodbye! Stay safe.")
                    self.running = False
                    break

                # --- THINKING ---
                self.state = AppState.THINKING
                self.conversation_history.append({
                    "role": "user",
                    "content": user_input
                })

                response = run_agentic_loop(
                    self.conversation_history,
                    self.tool_handlers
                )

                # --- SPEAKING ---
                self.state = AppState.SPEAKING
                speak(response)

            except KeyboardInterrupt:
                print("\nShutting down...")
                self.running = False
            except Exception as e:
                print(f"Error in main loop: {e}")
                speak("I encountered an error. Let me try again.")

        # Cleanup
        self._stop_continuous_timer()
        release_camera()
```

---

### `main.py` — Entry Point

```python
"""
Third Eye - Entry point.
An AI navigation and surroundings assistant for visually impaired people.

Usage:
    python main.py                  # Normal mode (on-demand)
    python main.py --no-riva        # Use fallback ASR/TTS (no Riva)
"""
import argparse
import config
from orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(description="Third Eye - AI Vision Assistant")
    parser.add_argument(
        "--no-riva",
        action="store_true",
        help="Use fallback speech engines instead of NVIDIA Riva"
    )
    args = parser.parse_args()

    if args.no_riva:
        config.USE_RIVA_ASR = False
        config.USE_RIVA_TTS = False
        print("Using fallback speech engines (SpeechRecognition + pyttsx3)")

    print("=" * 50)
    print("  THIRD EYE - AI Vision Assistant")
    print("  Speak to navigate. Say 'exit' to quit.")
    print("=" * 50)
    print()

    orchestrator = Orchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
```

---

## Data Flow Diagrams

### Flow 1: On-Demand Query — "What's in front of me?"

```
1. User speaks: "What's in front of me?"
   |
2. [Microphone] --> sounddevice captures 16kHz PCM
   |
3. [Riva ASR] --> gRPC to grpc.nvcf.nvidia.com:443 --> transcript: "What's in front of me?"
   |
4. [Orchestrator] appends to conversation_history:
   {"role": "user", "content": "What's in front of me?"}
   |
5. [agent.run_agentic_loop()] --> Nemotron API call:
   POST https://integrate.api.nvidia.com/v1/chat/completions
   {
     "model": "nvidia/nemotron-3-super-120b-a12b",
     "messages": [system_prompt, ...history, user_message],
     "tools": [capture_and_describe, get_directions, toggle_continuous_mode, read_text],
     "tool_choice": "auto"
   }
   |
6. Nemotron responds with tool_call:
   {"tool_calls": [{"function": {"name": "capture_and_describe", "arguments": "{}"}}]}
   |
7. [Orchestrator dispatches] --> vision.capture_and_describe()
   |
   +---> cv2.VideoCapture(0).read() --> frame
   +---> cv2.resize(frame, (640, 480))
   +---> cv2.imencode('.jpg', frame, quality=60) --> JPEG bytes
   +---> base64.b64encode(jpeg_bytes)
   |
8. [Llama 3.2 Vision API call]:
   POST https://integrate.api.nvidia.com/v1/chat/completions
   {
     "model": "meta/llama-3.2-90b-vision-instruct",
     "messages": [{"role": "user", "content": [
       {"type": "text", "text": "<accessibility prompt>"},
       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
     ]}],
     "max_tokens": 512
   }
   |
   Returns: "Directly ahead is a paved sidewalk with a slight upward slope..."
   |
9. [agent.run_agentic_loop()] feeds tool result back to Nemotron:
   {"role": "tool", "tool_call_id": "...", "content": "<vision description>"}
   |
10. [Nemotron API call #2] --> synthesizes natural response:
    "In front of you there's a sidewalk going slightly uphill. The path is
     clear for about 20 feet. There's a fire hydrant on your right side."
    |
11. [Riva TTS] --> gRPC to grpc.nvcf.nvidia.com:443 --> PCM audio bytes
    |
12. [sounddevice] plays audio through speakers
    |
13. User hears the description
```

### Flow 2: Navigation — "Take me to the nearest coffee shop"

```
1. User: "Take me to the nearest coffee shop from 500 Howard Street"
   |
2. [ASR] --> transcript
   |
3. [Nemotron] --> tool_call: get_directions(destination="nearest coffee shop", origin="500 Howard Street")
   |
4. [navigation.get_directions()] --> Apple MapKit:
   - CLGeocoder.geocodeAddressString("nearest coffee shop") --> coordinates
   - CLGeocoder.geocodeAddressString("500 Howard Street") --> coordinates
   - MKDirections.calculate() --> MKRoute with steps
   |
   Returns: "Walking directions to nearest coffee shop:\n
             Total distance: about 800 feet\n
             Estimated time: about 4 minutes\n
             Step 1: Head north on Howard St for 300 feet\n
             Step 2: Turn right on 2nd St for 400 feet\n
             Step 3: Destination is on your left"
   |
5. [Nemotron synthesizes]: "I found a coffee shop about 4 minutes walk away.
    Head north on Howard Street for about 300 feet, then turn right on
    2nd Street. The coffee shop will be on your left after about 400 feet.
    Would you like me to turn on continuous mode to guide you?"
   |
6. [TTS] --> audio
```

### Flow 3: Continuous Mode

```
User: "Yes, turn on continuous mode"
  |
[Nemotron] --> tool_call: toggle_continuous_mode(enabled=true)
  |
[Orchestrator] starts background thread:
  |
  Every 5 seconds (in background thread):
    |
    1. capture_and_describe() --> current scene description
    2. Compare with self.last_scene_description
    3. Send to Nemotron: CONTINUOUS_MODE_PROMPT_TEMPLATE with both descriptions
    4. If Nemotron responds with anything other than "NO_UPDATE":
       - Pause mic (state = SPEAKING)
       - TTS speaks the alert
       - Resume mic (state = LISTENING)
    5. Update self.last_scene_description

  Meanwhile, user can speak at ANY time:
    - If user speaks, orchestrator handles it normally
    - Continuous thread skips captures while state != LISTENING
    - User can say "stop continuous mode" to disable
```

---

## Implementation Order — Two-Person Parallel Split

### Person A: Mac Camera + Audio Pipeline
**Focus**: Get the Mac hardware (webcam, mic, speakers) working end-to-end.

**Phase A1 (Hours 1-3)**: Webcam + Audio setup
- Create `audio_utils.py` — test mic recording + speaker playback with sounddevice
- Create `vision.py` (webcam part only) — `cv2.VideoCapture`, `capture_frame()`
- Test: capture a frame, save to disk, confirm camera permissions work
- Test: record 3 seconds of audio, play it back

**Phase A2 (Hours 3-6)**: Voice I/O with Riva
- Create `speech.py` — implement `_speak_riva()` first (TTS is easier to test)
- Then implement `_listen_riva()` (ASR streaming)
- Also implement `_listen_fallback()` and `_speak_fallback()` as safety net
- Test: speak into mic -> get transcript -> synthesize response -> hear it

**Phase A3 (Hours 6-8)**: Navigation
- Create `navigation.py` — Apple MapKit via PyObjC
- Test: `get_directions("Starbucks", "500 Howard St San Francisco")` -> printed steps
- If MapKit fails due to signing issues, implement Google Maps REST fallback

**Phase A4 (Hours 8-10)**: Wire into orchestrator shell
- Create skeleton `orchestrator.py` with state machine
- Wire: listen() -> print transcript -> speak("I heard you say: " + transcript)
- Ready for merge with Person B's AI pipeline

---

### Person B: NVIDIA AI Models (using test photos)
**Focus**: Get all NVIDIA API integrations working with static test images.

**Phase B1 (Hours 1-3)**: Config + Nemotron
- Create `config.py` with all constants
- Create `.env` with NVIDIA_API_KEY
- Create `prompts.py` with all prompts
- Create `agent.py` — Nemotron client, test basic chat (no tools yet)
- Test: `call_nemotron([{"role": "system", ...}, {"role": "user", "content": "Hello"}])` -> response

**Phase B2 (Hours 3-6)**: Vision model with test photos
- Add `analyze_frame()` to `vision.py` — Llama 3.2 Vision API
- Create `test_photos/` directory with 3-4 static images (street scene, indoor, sign)
- Use `capture_frame_from_file()` helper to load test photos
- Test: `analyze_frame(test_photo_bytes, VISION_DESCRIBE_PROMPT)` -> scene description
- Test: `analyze_frame(sign_photo_bytes, VISION_READ_TEXT_PROMPT)` -> text reading
- Tune prompts based on quality of descriptions

**Phase B3 (Hours 6-8)**: Tool calling + Agentic loop
- Add `TOOL_DEFINITIONS` to `agent.py`
- Implement `run_agentic_loop()` with tool dispatch
- Create mock tool handlers that return static results
- Test full agentic loop: "What's in front of me?" -> Nemotron -> tool_call -> mock result -> Nemotron synthesis
- Test: "Take me to Starbucks" -> tool_call -> mock directions -> synthesis
- Test: "Read that sign" -> tool_call -> mock text -> synthesis

**Phase B4 (Hours 8-10)**: Continuous mode logic
- Implement continuous mode timer in orchestrator
- Implement `CONTINUOUS_MODE_PROMPT_TEMPLATE` comparison logic
- Test with alternating test photos to simulate scene changes
- Verify Nemotron correctly returns "NO_UPDATE" for similar scenes and alerts for changes

---

### Merge (Hours 10-12): Integration
- Person A's hardware pipeline + Person B's AI pipeline
- Wire real `capture_frame()` into `capture_and_describe()` (replace test photo loading)
- Wire real `listen()` / `speak()` into orchestrator
- Wire real `get_directions()` into tool handlers
- Full end-to-end test: speak into mic -> ASR -> Nemotron -> Vision API -> TTS

### Polish (Hours 12-14)
- Error handling: try/except around all API calls, graceful fallback messages
- Test edge cases: empty speech, API timeout, webcam permission denied
- Create `demo/demo_script.md` with rehearsed interaction sequence
- Update `README.md` with setup instructions and demo guide

---

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Riva cloud API unavailable or rate-limited | No voice I/O | Fallback to `SpeechRecognition` (Google) + `pyttsx3` (macOS native). Both are zero-config. Set `--no-riva` flag. |
| Riva ASR streaming too complex for hackathon timeline | Blocks voice input | Start with push-to-talk (press Enter to start/stop recording). Upgrade to streaming later if time permits. |
| MapKit PyObjC requires code signing or NSRunLoop | No navigation | Fallback to Google Maps Directions REST API (`maps.googleapis.com/maps/api/directions/json`). Pre-configure a Google Maps API key. |
| Llama Vision API latency (2-5 seconds per frame) | Slow responses | Compress images (640x480, JPEG quality 60). Run in background thread. Play "Let me take a look..." via TTS while waiting. |
| Nemotron tool calling unreliable | Wrong/missing tool calls | Write very explicit tool descriptions. Add keyword-based forced dispatch in orchestrator (if user says "see"/"look"/"front" and Nemotron didn't call capture_and_describe, force it). |
| Audio feedback loops (speaker -> mic) | Garbled audio | State machine: mic is muted/paused during SPEAKING state. Only one audio direction active at a time. |
| Continuous mode burns through API quota | Rate limiting | Default interval 5-8 seconds. Skip VILA call if frame hash is very similar to previous (user is stationary). Cache last description. |
| PyObjC event loop conflicts with threading | Crashes/hangs | Keep app synchronous with threading (not asyncio). Run MapKit on main thread. Use daemon threads for background work. |

---

## Environment Setup Script

```bash
#!/bin/bash
# setup.sh - Run this first on a fresh Mac

# 1. Install system dependencies
brew install portaudio
xcode-select --install  # if not already installed

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Create .env file
cp .env.example .env
echo "Now edit .env and add your NVIDIA_API_KEY"

# 5. Test webcam permission
python -c "import cv2; cam = cv2.VideoCapture(0); print('Camera OK' if cam.isOpened() else 'Camera FAILED'); cam.release()"

# 6. Test audio
python -c "import sounddevice; print(sounddevice.query_devices())"

# 7. Run the app
python main.py
```

---

## .env.example

```
NVIDIA_API_KEY=nvapi-your-key-here
```

---

## .gitignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
venv/
.venv/
*.egg-info/

# Environment
.env

# macOS
.DS_Store

# IDE
.idea/
.vscode/
*.swp

# Test artifacts
test_photos/*.result.txt
```

---

## Verification Plan

1. **Module-level tests** (during development):
   - `python -c "from config import NVIDIA_API_KEY; print('Config OK')"` — verify .env loading
   - `python -c "from agent import call_nemotron, create_initial_history; h = create_initial_history(); h.append({'role':'user','content':'Hello'}); print(call_nemotron(h).message.content)"` — verify Nemotron
   - `python -c "from vision import capture_frame; f = capture_frame(); print(f'Frame: {len(f)} bytes')"` — verify webcam
   - `python -c "from vision import analyze_frame; print(analyze_frame(open('test_photos/street.jpg','rb').read(), 'Describe this.'))"` — verify Llama Vision
   - `python -c "from speech import speak; speak('Hello, I am Third Eye.')"` — verify TTS

2. **Integration smoke test**: `python main.py` → speak "Hello" → verify you hear a response

3. **Vision test**: Say "What do you see?" pointing webcam at a room → verify accurate description

4. **Navigation test**: Say "How do I get to [local landmark] from [your address]?" → verify step-by-step directions

5. **Continuous mode test**: Say "Turn on continuous mode" → move laptop around → verify proactive alerts → say "Stop continuous mode"

6. **Full demo rehearsal**: Run through the scripted demo sequence in `demo/demo_script.md`
