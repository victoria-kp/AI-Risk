# AI Risk

## Overview

End-to-end pipeline to fine-tune [Qwen 2.5 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) to play the board game [Risk](https://en.wikipedia.org/wiki/Risk_(game)). The game environment is [PyRisk](https://github.com/reiddraper/pyrern) (vendored in `pyrisk_vendor/`).

In Risk, players place troops across 42 territories on a world map. Each turn has three phases:

1. **Reinforce** -- place new troops on owned territories.
2. **Attack** -- attack adjacent enemy territories (battles resolved by dice).
3. **Move** -- relocate troops between owned adjacent territories.

Owning all territories in a continent grants bonus troops each turn. The player who conquers all 42 territories wins.

PyRisk includes four built-in bots: **StupidAI** (random actions), **BetterAI**, **AlAI**, and **ChronAI** (heuristic strategies such as prioritizing smaller continents for the bonus).

## Architecture

The Qwen model is trained to make two types of decisions:

1. **Reinforcements** -- where to place available troops.
2. **Attacks** -- which territories to attack from a menu of valid options.

Initial placement and troop movement are handled by BetterAI heuristics (see `llm_player/hybrid_player.py`). The model receives a structured prompt with the board state, its territories, and available options, and outputs a short reasoning followed by JSON.

**Reinforcement example:**

```json
{"reinforcements": {"Southern Europe": 5, "Ukraine": 3}}
```

**Attack example** (indices from a numbered menu of valid attacks):

```json
{"attacks": [2, 3]}
```

If the model's output fails to parse, the system falls back to BetterAI for that decision.

## Training

Training was done on an H100 GPU via Google Colab (see `training/AI-Risk-Training.ipynb`). The pipeline has three stages:

### 1. Data Collection

Heuristic data was collected by running 500 BetterAI and 500 ChronAI games using `data/collect_heuristic_data.py`, producing ~27,000 turn entries in `data/hybrid_data/turns.jsonl`.

### 2. Supervised Fine-Tuning (SFT)

The base Qwen 2.5 7B model was fine-tuned on the heuristic data to learn the correct JSON output format.

- **Script:** `training/train_sft.py`
- **Steps:** 200
- **Learning rate:** 2e-5
- **Batch size:** 1 (gradient accumulation: 4)
- **Quantization:** QLoRA (4-bit NF4 via bitsandbytes)
- **Validation:** 5% split, eval every 50 steps

### 3. Group Relative Policy Optimization (GRPO)

After SFT, the model was trained with GRPO to improve strategic decision-making using a custom reward function.

**Round 1 (GRPO v2)** -- trained on heuristic data:
- **Script:** `training/train_grpo.py`
- **Steps:** 50
- **Learning rate:** 1e-6
- **Generations per prompt:** 4

**Round 2 (GRPO v3)** -- trained on heuristic data + on-policy benchmark data:
- **Resume from:** GRPO v2
- **Data:** `data/hybrid_data/turns.jsonl` + `results/qwen_grpo_v2_benchmark/turns.jsonl`
- **Steps:** 50
- **Learning rate:** 1e-6
- **Generations per prompt:** 8

The final trained model is available on Hugging Face: [victoria-kp/risk-qwen-grpo-v3](https://huggingface.co/victoria-kp/risk-qwen-grpo-v3)

### Reward Function

The reward function (`training/reward_hybrid.py`) scores each decision with **25% format / 75% strategy** weighting:

**Reinforcements:**

| Component | Weight | Description |
|---|---|---|
| JSON validity | 0.15 | Valid JSON with `"reinforcements"` dict |
| Correctness | 0.10 | Troops sum to available, all territories owned |
| Concentration | 0.30 | Penalize spreading across 4+ territories |
| Border placement | 0.25 | Fraction of troops placed on border territories |
| Continent completion | 0.20 | Reward reinforcing near completable continents |

**Attacks:**

| Component | Weight | Description |
|---|---|---|
| JSON validity | 0.15 | Valid JSON with `"attacks"` list |
| Index validity | 0.10 | All indices in valid range |
| Attack quality | 0.30 | Troop ratio (attacker vs defender) |
| Activity bonus | 0.15 | Reward attacking when favorable options exist |
| Continent targeting | 0.30 | Reward attacks that complete or deny continents |

Since SFT already teaches the model to produce correct JSON, the reward function weights strategy much more heavily than format.

### Weights & Biases

Training metrics were synced to W&B after the training runs to confirm the logs in the Jupyter notebook: [W&B Project Dashboard](https://wandb.ai/1victoriakp-1/AI-Risk/overview)

The KL divergence in the first GRPO output (GRPO v1) was large, indicating the policy drifted significantly from the reference model. In GRPO v3, the KL divergence is under control, showing the model improved its strategy without diverging too far from the base policy.

## Results

Results obtained by running `python analysis/compare_results.py --reward`. Each model played 20 games against 2 StupidAI opponents.

| Model | Win Rate | Fallbacks | LLM Reward | Reinforce Reward | Attack Reward |
|---|---|---|---|---|---|
| Gemini Flash Lite | 0% | 1% | 0.777 | 0.706 | 0.835 |
| Qwen 7B Base | 5% | 14% | 0.774 | 0.602 | 0.848 |
| Qwen SFT | 15% | 5% | 0.813 | 0.729 | 0.860 |
| Qwen GRPO v2 | 30% | 7% | 0.796 | 0.685 | 0.864 |
| **Qwen GRPO v3** | **30%** | **5%** | **0.813** | **0.731** | **0.868** |

There is a clear progression in win rates, fallback rates, and reward scores across training stages. While the 30% win rate is roughly what a random player (StupidAI) would achieve against two other random players (~33%), the model shows measurable improvement at each stage and outperforms Gemini Flash Lite out of the box.

### Future Improvements

- **More on-policy iterations** -- continue the play-train-play loop for additional rounds
- **More GRPO steps** -- 50 steps per round is conservative; 200-500 with lower LR (5e-7) could help
- **Larger training data** -- 50-100 benchmark games per round instead of 20
- **Larger model** -- Qwen 14B or 72B for better board state reasoning
- **Full-phase LLM control** -- let the LLM handle placement and movement, removing BetterAI dependency
- **Reward tuning** -- add troop economy scoring (troops gained vs lost per attack)

## Playing

You can play against any of the PyRisk bots, Gemini Flash Lite, or the fine-tuned Qwen model using the interactive notebook at `playing/interactive_game.ipynb`.

- **Gemini:** requires a Google API key set as `GOOGLE_API_KEY`
- **Qwen GRPO v3:** downloads the model from [Hugging Face](https://huggingface.co/victoria-kp/risk-qwen-grpo-v3) (requires GPU + ~16GB VRAM)

## Requirements

**Local (playing + inference):**
```
google-generativeai
transformers
peft
torch
```

**Training (Colab):**
```
trl==0.14.0
peft
datasets
bitsandbytes
```

## Setup

```bash
git clone https://github.com/victoria-kp/AI-Risk.git
cd AI-Risk
pip install -r requirements.txt
```

To run a benchmark:
```bash
# Gemini (needs GOOGLE_API_KEY)
python analysis/run_benchmark.py --games 20 --output results/gemini_benchmark

# Fine-tuned Qwen (needs GPU + peft installed)
# First download the adapter:
git clone https://huggingface.co/victoria-kp/risk-qwen-grpo-v3
# Then run:
python analysis/run_benchmark.py --games 20 --model ./risk-qwen-grpo-v3 --output results/grpo_v3_benchmark
```

To compare results:
```bash
python analysis/compare_results.py --reward
```

## Project Structure

```
AI-Risk/
├── llm_player/           # Hybrid player + model backend
│   ├── hybrid_player.py  # BetterAI + LLM hybrid
│   ├── decision_menus.py # Prompt building + parsing
│   └── model.py          # Model backend (PEFT/Transformers/Gemini)
├── training/
│   ├── train_sft.py      # Supervised fine-tuning
│   ├── train_grpo.py     # GRPO reinforcement learning
│   ├── reward_hybrid.py  # Custom reward function
│   └── AI-Risk-Training.ipynb  # Colab training notebook
├── analysis/
│   ├── run_benchmark.py  # Run games and save turn logs
│   ├── compare_results.py # Compare all benchmarks
│   ├── score_turns.py    # Score turns with reward function
│   └── evaluate.py       # Detailed decision metrics
├── data/
│   ├── collect_heuristic_data.py  # Generate training data
│   └── hybrid_data/      # Training data (turns.jsonl)
├── results/              # Benchmark results per model
├── playing/
│   └── interactive_game.ipynb  # Play against any bot
├── pyrisk_vendor/        # PyRisk game engine
└── requirements.txt
```

## Contact

Built by Victoria Knapp Perez ([1victoriakp@gmail.com](mailto:1victoriakp@gmail.com)).
