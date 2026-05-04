#!/bin/bash
# Master pipeline: runs all experiments from the proposal
# Usage: bash run_all.sh [--quick]
#   --quick: use --max_n 100 / --n_samples 50 and skip slow detectors for testing

set -e
mkdir -p results

MAX_N_FLAG=""
SKIP_FLAGS=""
N_SAMPLES=500

if [ "$1" = "--quick" ]; then
    MAX_N_FLAG="--max_n 100"
    SKIP_FLAGS="--skip_detectgpt --skip_dnagpt"
    N_SAMPLES=50
    echo ">>> QUICK MODE: 100 samples, fast detectors only"
fi

echo ""
echo "================================================================"
echo "Section 2.1 — Figure 1: ChatGPT Detection (Ghostbuster)"
echo "================================================================"
python run_fig1.py --out results $MAX_N_FLAG $SKIP_FLAGS

echo ""
echo "================================================================"
echo "Section 2.3 — Figure 2: Document Length Ablation"
echo "================================================================"
python run_fig2.py --out results $MAX_N_FLAG

echo ""
echo "================================================================"
echo "Section 2.2 — Figure 3: LLaMA-2-13B ROC Curves"
echo "================================================================"
python run_fig3.py --out results $MAX_N_FLAG $SKIP_FLAGS

echo ""
echo "================================================================"
echo "Section 2.4 — Figure 7: Non-Native Speaker Robustness"
echo "================================================================"
python run_fig7.py --out results $MAX_N_FLAG

echo ""
echo "================================================================"
echo "Section 2.5 — Figure 8: Modified Prompting Strategies"
echo "================================================================"
python run_fig8.py --out results --n_samples $N_SAMPLES

echo ""
echo "================================================================"
echo "Section 3.1 — Extension 1: Newer LLMs (Mistral-7B, Llama-3-8B)"
echo "================================================================"
python run_ext1.py --out results --n_samples $N_SAMPLES

echo ""
echo "================================================================"
echo "Generating All Plots"
echo "================================================================"
python plot_all.py --dir results

echo ""
echo "================================================================"
echo "ALL DONE — Results in results/"
echo "================================================================"
ls -la results/
