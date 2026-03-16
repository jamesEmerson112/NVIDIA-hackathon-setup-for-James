#!/usr/bin/env bash
set -euo pipefail

# Tier 1: NVIDIA NIM API Setup
# Zero GPU, zero cost — uses the free NIM API at build.nvidia.com
# Model: nvidia/nemotron-3-super-120b

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Tier 1: NVIDIA NIM API Setup ==="
echo ""

# --- Check Python 3 ---
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.8+."
    exit 1
fi
echo "Found python3: $(python3 --version)"

# --- Install openai package ---
echo ""
echo "Installing openai package..."
pip install --quiet openai
echo "openai package installed."

# --- Get API key ---
echo ""
if [ -z "${NIM_API_KEY:-}" ]; then
    echo "No NIM_API_KEY environment variable found."
    echo "Get your free API key at: https://build.nvidia.com"
    echo ""
    read -rp "Enter your NIM API key (nvapi-...): " NIM_API_KEY
    if [ -z "$NIM_API_KEY" ]; then
        echo "ERROR: No API key provided."
        exit 1
    fi
fi

# --- Create test script ---
TEST_SCRIPT="$SCRIPT_DIR/tier1_test.py"
cat > "$TEST_SCRIPT" << 'PYEOF'
import os
import sys

from openai import OpenAI

api_key = os.environ.get("NIM_API_KEY")
if not api_key:
    print("ERROR: NIM_API_KEY not set.")
    sys.exit(1)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
)

print("Sending test request to Nemotron Super 120B...")
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
    print("SUCCESS: NIM API is working.")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
PYEOF

echo ""
echo "Running test call..."
echo ""
NIM_API_KEY="$NIM_API_KEY" python3 "$TEST_SCRIPT"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Export your key:  export NIM_API_KEY='$NIM_API_KEY'"
echo "  2. Build your app using the OpenAI-compatible client (see tier1_test.py)"
echo "  3. Base URL: https://integrate.api.nvidia.com/v1"
echo "  4. Model:    nvidia/nemotron-3-super-120b"
