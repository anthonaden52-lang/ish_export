#!/bin/sh
# Quick setup script for iSH (Alpine)

set -e

echo "=== iSH Bot Setup ==="

# System packages
apk update
apk add --no-cache python3 py3-pip python3-dev gcc g++ musl-dev libffi-dev openssl-dev wget unzip

# Virtualenv
python3 -m venv .venv
. .venv/bin/activate

pip install --upgrade pip

echo "Installing core packages..."
pip install requests pandas numpy joblib python-dotenv

echo ""
echo "Trying scikit-learn (may fail or take forever on iPhone)..."
pip install scikit-learn || echo "scikit-learn failed — training will not work, but scanner/paper trader still can."

echo ""
echo "Trying solders (needed for real swaps)..."
pip install solders || echo "solders failed — real --execute swaps won't work on iSH."

echo ""
echo "=== Done ==="
echo "Activate with:  source .venv/bin/activate"
echo "Then try:       python3 swap_bot.py --from SOL --to USDC --amount 0.01"
echo ""
echo "Read ISH_SETUP.md for full instructions."
