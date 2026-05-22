#!/bin/bash
# Deploy FBC to RunPod GPU instance and setup Mamba-SSM
# Usage: ./deploy_to_runpod.sh <pod-host> [ssh-key-path]
# Example: ./deploy_to_runpod.sh p4u20qptqv96qz-64411fc9@ssh.runpod.io ~/.ssh/id_ed25519

set -e

POD_HOST="${1:-}"
SSH_KEY="${2:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="/workspace/fbc-core"

if [ -z "$POD_HOST" ]; then
    echo "Usage: $0 <pod-host> [ssh-key-path]"
    echo "Example: $0 p4u20qptqv96qz-64411fc9@ssh.runpod.io"
    exit 1
fi

echo "=== Deploying FBC to RunPod ==="
echo "Host: $POD_HOST"
echo "SSH Key: $SSH_KEY"

# 1. Create remote directory
echo ""
echo "[1/5] Creating remote workspace..."
ssh -i "$SSH_KEY" "$POD_HOST" "mkdir -p $REMOTE_DIR"

# 2. Sync code (excluding local dev files)
echo ""
echo "[2/5] Syncing FBC code..."
rsync -avz --progress \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='*.egg-info' \
    --exclude='dev-docs/' \
    --exclude='demos/' \
    --exclude='benchmarks/' \
    -e "ssh -i $SSH_KEY" \
    "$(dirname "$0")/../" \
    "$POD_HOST:$REMOTE_DIR/"

# 3. Run GPU setup on remote
echo ""
echo "[3/5] Running Mamba-SSM GPU setup (this takes 5-10 minutes)..."
ssh -i "$SSH_KEY" "$POD_HOST" "cd $REMOTE_DIR && chmod +x scripts/setup_mamba_gpu.sh && ./scripts/setup_mamba_gpu.sh"

# 4. Verify installation
echo ""
echo "[4/5] Verifying Mamba-SSM installation..."
ssh -i "$SSH_KEY" "$POD_HOST" "cd $REMOTE_DIR && python3 scripts/verify_mamba_gpu.py"

# 5. Run tests
echo ""
echo "[5/5] Running FBC tests with Mamba-3..."
ssh -i "$SSH_KEY" "$POD_HOST" "cd $REMOTE_DIR && python3 -m pytest tests/test_text_modality.py tests/test_pipeline_e2e.py -v --tb=short 2>&1 | tail -20"

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "To use FBC with Mamba-3 on the cloud instance:"
echo "  ssh -i $SSH_KEY $POD_HOST"
echo "  cd $REMOTE_DIR"
echo "  python3 -c \"from fbc.pipeline import FBCPipeline; p = FBCPipeline(use_mamba=True).cuda()\""
