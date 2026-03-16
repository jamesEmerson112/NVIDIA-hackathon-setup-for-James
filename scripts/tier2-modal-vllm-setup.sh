#!/usr/bin/env bash
set -euo pipefail

# Tier 2: Modal + vLLM Serverless Setup
# Serves Nemotron Super 120B on a cloud GPU via Modal
# Free $30/mo credits — no credit card required to start

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Tier 2: Modal + vLLM Serverless Setup ==="
echo ""

# --- Check Python 3 ---
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.8+."
    exit 1
fi
echo "Found python3: $(python3 --version)"

# --- Install packages ---
echo ""
echo "Installing modal and openai packages..."
pip install --quiet modal openai
echo "Packages installed."

# --- Check Modal auth ---
echo ""
if ! modal token list &>/dev/null 2>&1; then
    echo "Modal not authenticated. Running 'modal setup'..."
    echo "This will open a browser to log in (free account, \$30/mo GPU credits)."
    echo ""
    modal setup
else
    echo "Modal already authenticated."
fi

# --- Check for HuggingFace token ---
echo ""
if [ -z "${HF_TOKEN:-}" ]; then
    echo "NOTE: Nemotron models may require a HuggingFace token for download."
    echo "If needed, set HF_TOKEN or create a Modal secret named 'huggingface':"
    echo "  modal secret create huggingface HF_TOKEN=hf_xxx"
    echo ""
fi

# --- Create Modal deployment file ---
DEPLOY_FILE="$SCRIPT_DIR/tier2_modal_vllm.py"
cat > "$DEPLOY_FILE" << 'PYEOF'
"""Modal + vLLM deployment for Nemotron Super 120B.

Usage:
    modal serve tier2_modal_vllm.py    # dev mode (live reload, temporary URL)
    modal deploy tier2_modal_vllm.py   # production (persistent URL)
"""

import subprocess

import modal

MODEL_NAME = "nvidia/nemotron-3-super-120b"
VLLM_PORT = 8000

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("nemotron-vllm")


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
@modal.web_server(port=VLLM_PORT)
def serve():
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]
    subprocess.Popen(cmd)
PYEOF

echo "Created: $DEPLOY_FILE"

# --- Create test client script ---
TEST_SCRIPT="$SCRIPT_DIR/tier2_test.py"
cat > "$TEST_SCRIPT" << 'PYEOF'
"""Test client for the Modal vLLM deployment.

Usage:
    python tier2_test.py <YOUR_MODAL_URL>

The URL is printed when you run `modal serve` or `modal deploy`.
"""

import sys

from openai import OpenAI

if len(sys.argv) < 2:
    print("Usage: python tier2_test.py <MODAL_ENDPOINT_URL>")
    print("  e.g. python tier2_test.py https://your-workspace--nemotron-vllm-serve.modal.run")
    sys.exit(1)

base_url = sys.argv[1].rstrip("/") + "/v1"

client = OpenAI(base_url=base_url, api_key="not-needed")

print(f"Sending test request to {base_url} ...")
print("")

try:
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b",
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        max_tokens=64,
    )
    print("Model response:")
    print(response.choices[0].message.content)
    print("")
    print("SUCCESS: Modal vLLM endpoint is working.")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
PYEOF

echo "Created: $TEST_SCRIPT"

# --- Print next steps ---
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Files created:"
echo "  $DEPLOY_FILE    — Modal deployment (vLLM + Nemotron)"
echo "  $TEST_SCRIPT       — Test client"
echo ""
echo "Next steps:"
echo ""
echo "  1. (If needed) Create a HuggingFace secret:"
echo "     modal secret create huggingface HF_TOKEN=hf_xxx"
echo ""
echo "  2. Start the server in dev mode:"
echo "     cd $SCRIPT_DIR && modal serve tier2_modal_vllm.py"
echo ""
echo "  3. Copy the URL printed by Modal, then test:"
echo "     python3 $TEST_SCRIPT https://your-workspace--nemotron-vllm-serve.modal.run"
echo ""
echo "  4. For a persistent deployment:"
echo "     cd $SCRIPT_DIR && modal deploy tier2_modal_vllm.py"
echo ""
echo "Cost: Free \$30/mo credits. H100 usage billed per-second only while serving."
