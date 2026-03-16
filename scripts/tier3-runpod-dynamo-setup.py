#!/usr/bin/env python3
"""Tier 3: NVIDIA Dynamo on RunPod — Disaggregated Prefill/Decode

Runs from your laptop. Provisions a 2x A100 80GB pod on RunPod,
deploys Dynamo with SGLang (disaggregated prefill/decode) serving
Nemotron Super 120B, and returns an OpenAI-compatible endpoint URL.

Prerequisites:
    pip install runpod openai

Usage:
    python3 tier3-runpod-dynamo-setup.py

Environment variables (or prompted):
    RUNPOD_API_KEY  — from https://runpod.io/console/user/settings
    HF_TOKEN        — from https://huggingface.co/settings/tokens
"""

import os
import sys
import time

MODEL_NAME = "nvidia/nemotron-3-super-120b"
DYNAMO_IMAGE = "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.0"
GPU_TYPE = "NVIDIA A100 80GB"
GPU_COUNT = 2
EXPOSED_PORT = 8000

# Startup script that runs inside the pod.
# Launches all 3 Dynamo processes: frontend, prefill worker, decode worker.
STARTUP_SCRIPT = f"""#!/bin/bash
set -e

echo "=== Starting Dynamo (disaggregated prefill/decode) ==="

# Launch Dynamo frontend (API gateway) on port {EXPOSED_PORT}
python3 -m dynamo.frontend --http-port {EXPOSED_PORT} --discovery-backend file &
FRONTEND_PID=$!
echo "Frontend started (PID $FRONTEND_PID)"

# Give frontend a moment to bind
sleep 3

# Launch prefill worker on GPU 0
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.sglang \\
    --model-path {MODEL_NAME} \\
    --discovery-backend file \\
    --disaggregation-mode prefill &
PREFILL_PID=$!
echo "Prefill worker started on GPU 0 (PID $PREFILL_PID)"

# Launch decode worker on GPU 1
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.sglang \\
    --model-path {MODEL_NAME} \\
    --discovery-backend file \\
    --disaggregation-mode decode &
DECODE_PID=$!
echo "Decode worker started on GPU 1 (PID $DECODE_PID)"

echo "=== All Dynamo processes launched ==="
echo "Waiting for model download and warmup..."

# Keep container alive
wait
"""


def get_env_or_prompt(var_name, prompt_text, sensitive=True):
    """Read from environment or prompt the user."""
    value = os.environ.get(var_name, "").strip()
    if value:
        masked = value[:8] + "..." if sensitive and len(value) > 8 else value
        print(f"  {var_name} found in environment ({masked})")
        return value
    value = input(f"  {prompt_text}: ").strip()
    if not value:
        print(f"ERROR: {var_name} is required.")
        sys.exit(1)
    return value


def wait_for_pod_running(pod_id, max_wait=300):
    """Poll until the pod reaches RUNNING state."""
    import runpod

    start = time.time()
    while time.time() - start < max_wait:
        pod = runpod.get_pod(pod_id)
        status = pod.get("desiredStatus", "UNKNOWN")
        runtime = pod.get("runtime")
        if runtime and runtime.get("uptimeInSeconds"):
            return pod
        elapsed = int(time.time() - start)
        print(f"  Pod status: {status} ({elapsed}s elapsed)")
        time.sleep(10)
    print(f"ERROR: Pod did not reach RUNNING state within {max_wait}s.")
    sys.exit(1)


def wait_for_endpoint(base_url, max_wait=600):
    """Poll the /v1/models endpoint until it responds."""
    import urllib.request
    import urllib.error

    url = f"{base_url}/v1/models"
    start = time.time()
    while time.time() - start < max_wait:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        elapsed = int(time.time() - start)
        print(f"  Waiting for Dynamo endpoint... ({elapsed}s elapsed)")
        time.sleep(15)
    return False


def run_test_call(base_url):
    """Send a test completion request via the OpenAI client."""
    from openai import OpenAI

    client = OpenAI(base_url=f"{base_url}/v1", api_key="not-needed")
    print("\nSending test request to Nemotron Super 120B via Dynamo...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        max_tokens=64,
    )
    print(f"Model response: {response.choices[0].message.content}")
    print("\nSUCCESS: Dynamo endpoint is working.")


def main():
    print("=== Tier 3: NVIDIA Dynamo on RunPod ===\n")

    # --- Check dependencies ---
    try:
        import runpod  # noqa: F401
    except ImportError:
        print("Installing runpod and openai packages...")
        os.system(f"{sys.executable} -m pip install --quiet runpod openai")
        import runpod  # noqa: F401

    from openai import OpenAI  # noqa: F401

    # --- Get credentials ---
    print("Credentials:")
    runpod_key = get_env_or_prompt(
        "RUNPOD_API_KEY",
        "Enter your RunPod API key (from runpod.io/console/user/settings)",
    )
    hf_token = get_env_or_prompt(
        "HF_TOKEN",
        "Enter your HuggingFace token (from huggingface.co/settings/tokens)",
    )

    import runpod

    runpod.api_key = runpod_key

    # --- Create pod ---
    print(f"\nCreating RunPod: {GPU_COUNT}x {GPU_TYPE}, image={DYNAMO_IMAGE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Estimated cost: ~${GPU_COUNT * 1.20:.2f}/hr\n")

    pod = runpod.create_pod(
        name="dynamo-nemotron",
        image_name=DYNAMO_IMAGE,
        gpu_type_id=GPU_TYPE,
        gpu_count=GPU_COUNT,
        container_disk_in_gb=50,
        volume_in_gb=100,
        ports=f"{EXPOSED_PORT}/http",
        env={
            "HF_TOKEN": hf_token,
            "HUGGING_FACE_HUB_TOKEN": hf_token,
        },
        docker_args=STARTUP_SCRIPT,
    )

    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")

    # --- Wait for pod ---
    print("\nWaiting for pod to start...")
    wait_for_pod_running(pod_id)
    print("Pod is running!")

    base_url = f"https://{pod_id}-{EXPOSED_PORT}.proxy.runpod.net"
    print(f"Endpoint URL: {base_url}")

    # --- Wait for Dynamo to be ready ---
    print("\nWaiting for Dynamo to download model and start serving...")
    print("(This can take 5-15 minutes for the first run as it downloads the model)")
    if not wait_for_endpoint(base_url, max_wait=900):
        print("\nWARNING: Endpoint not responding yet.")
        print("The model may still be downloading. Try again in a few minutes:")
        print(f"  curl {base_url}/v1/models")
        print(f"\nPod ID: {pod_id}")
        print(f"To terminate: python3 -c \"import runpod; runpod.api_key='{runpod_key}'; runpod.terminate_pod('{pod_id}')\"")
        sys.exit(1)

    # --- Test call ---
    run_test_call(base_url)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("DYNAMO ENDPOINT READY")
    print("=" * 60)
    print(f"  Endpoint:  {base_url}/v1")
    print(f"  Model:     {MODEL_NAME}")
    print(f"  Setup:     Disaggregated prefill/decode on {GPU_COUNT}x {GPU_TYPE}")
    print(f"  Pod ID:    {pod_id}")
    print(f"  Cost:      ~${GPU_COUNT * 1.20:.2f}/hr (billed per second)")
    print()
    print("Use with the OpenAI client:")
    print(f'  client = OpenAI(base_url="{base_url}/v1", api_key="not-needed")')
    print()
    print("IMPORTANT — Terminate when done to stop billing:")
    print(f"  python3 -c \"import runpod; runpod.api_key='{runpod_key}'; runpod.terminate_pod('{pod_id}')\"")
    print()


if __name__ == "__main__":
    main()
