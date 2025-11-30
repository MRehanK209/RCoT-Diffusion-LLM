#!/bin/bash
# Quick test script for diffusion models
# This script runs a quick evaluation on a small subset to verify models work correctly

set -e  # Exit on error

echo "======================================"
echo "Testing Diffusion Models Integration"
echo "======================================"
echo ""

# Create results directory
mkdir -p results/quick_tests

# Test 1: Dream 7B
echo "Test 1: Dream 7B on 5 GSM8K problems"
echo "--------------------------------------"
python evaluate_passk.py \
  --model_name Dream-org/Dream-v0-Instruct-7B \
  --dataset gsm8k \
  --n_samples 10 \
  --subset_size 5 \
  --temperatures 0.2 \
  --diffusion_steps 256 \
  --diffusion_alg entropy \
  --output_dir results/quick_tests \
  --verbose