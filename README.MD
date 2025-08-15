# LLM Evaluation Framework

This repository contains a modular framework for evaluating large language models (LLMs) in alignment with human judgments, while mitigating evaluation biases and reducing computational costs.

## Overview

The framework implements several advanced evaluation methodologies:

- **Fair Evaluation (MEC+BPC)**: Implements unbiased evaluation protocols with Multi-Expert Consensus and Position Balancing to mitigate position and order biases
- **Cascaded Evaluation**: Combines multiple evaluation signals for more robust assessment
- **Calibrated Judge (CJ)**: Uses model calibration and uncertainty measures to weight evaluation judgments
- **Calibrated Scorer (CS)**: Addresses verbosity bias in evaluation metrics through length-controlled methodology

## Codebase

The core evaluation components are located in the `code_components/` directory:

### Core Evaluation Modules

- **`fair_eval.py`**: Implements fair evaluation protocols with MEC+BPC methodology
  - **MEC (Multi-Expert Consensus)**: Aggregates multiple judge opinions for robust evaluation
  - **BPC (Position Balancing)**: Mitigates position bias through strategic ordering
  - Supports multiple evaluation models (GPT-3.5, GPT-4)
  - Implements k-sampling and position balancing
  - Includes cost tracking for API usage

- **`cascaded_eval.py`**: Applies the cascaded evaluation methodology
  - Combines reliability scores with human judgments for optimal assessment
  - Implements intelligent routing based on reliability rankings
  - Handles forward/reverse order averaging for bias reduction
  - Merges multiple evaluation signals for robust outcomes

- **`cal_reliability.py`**: Generates responses from fine-tuned models and calculates reliability scores
  - Uses VLLM for efficient generation of multiple responses
  - Computes uncertainty-based reliability metrics (entropy, variance, logits)
  - Supports both auto-j and judgelm evaluation types
  - Generates reliability scores for cascaded evaluation

- **`lc_component.py`**: Implements the Calibrated Scorer (CS) component
  - Addresses verbosity bias through regression-based debiasing
  - Uses cross-validation for robust parameter estimation
  - Maintains interpretability while controlling for length effects
  - Implements Length-Controlled AlpacaEval methodology

### Supporting Components

- **`evaluate_judge.py`**: Utility functions for judge evaluation
- **`build_dataset.py`**: Dataset construction and preprocessing utilities
- **`build_prompt_judge.py`**: Prompt engineering for evaluation tasks

## Data(sets) description

The framework supports multiple datasets and evaluation protocols:

### Alpaca Datasets
Located in `datasets/alpaca/`, containing evaluations for various models:
- **Base Models**: alpaca-7b, claude-2.1, gpt-3.5-turbo, gpt4, Mixtral-8x7B
- **Variants**: Concise and verbose versions for verbosity analysis
- **Evaluation Types**: Standard AlpacaEval, CoT-based evaluation, weighted evaluation

### Vicuna Datasets
Located in `datasets/vicuna/`, featuring:
- **Question Sets**: Diverse instruction-following tasks
- **Model Responses**: Multiple model outputs for comparison
- **Sampled Data**: Subsampled datasets according to the sampling methodology
- **Repeated Sampling**: Multiple sampling runs for statistical robustness

### Experimental Data
Located in `gathered_data/`, containing:
- **Baseline Results**: Standard evaluation outcomes
- **MEC+BPC Experiments**: Multi-expert consensus and position balancing results
- **Cascaded Evaluation Results**: Combined reliability and judgment outcomes
- **Expanded Reviews**: Detailed evaluation analyses

## Repository structure

```
thesis_code/
├── code_components/          # Core evaluation algorithms
│   ├── fair_eval.py         # MEC+BPC fair evaluation implementation
│   ├── cascaded_eval.py     # Cascaded evaluation methodology
│   ├── cal_reliability.py   # Calibrated Judge (CJ) - reliability scoring
│   ├── lc_component.py      # Calibrated Scorer (CS) - length-controlled evaluation
│   └── ...                  # Supporting utilities
├── datasets/                 # Evaluation datasets
│   ├── alpaca/              # AlpacaEval datasets
│   └── vicuna/              # Vicuna instruction datasets
├── gathered_data/            # Experimental results
│   ├── baseline_mec_results/ # Baseline and MEC experiments
│   ├── cascaded-eval-results/ # Cascaded evaluation outcomes
│   └── expanded_review/      # Detailed analysis results
├── analysis/                 # Analysis notebooks
│   ├── experimental_result_analysis.ipynb  # Main results analysis
│   └── dataset_sampling.ipynb             # Data sampling methodology
├── scripts/                  # Experiment automation
│   ├── cj_test.sh           # Main experimental pipeline
│   ├── cj_temperature_test.sh # Temperature sensitivity tests
│   └── cj_cs_test.sh        # Cross-validation tests
├── models/                   # Local model storage
├── figures/                  # Generated visualizations
└── requirements.txt          # Python dependencies
```

## Analysis notebook

The `analysis/` directory contains comprehensive analysis notebooks:

- **`experimental_result_analysis.ipynb`**: Main analysis of experimental results
  - Statistical analysis of evaluation outcomes
  - Comparison of different evaluation methodologies
  - Bias mitigation effectiveness assessment
  - Performance correlation analysis

- **`dataset_sampling.ipynb`**: Implements the data sampling methodology
  - Sampling strategy implementation
  - Statistical power analysis
  - Representative subset selection
  - Sampling efficiency evaluation

## Experiments replication

The framework includes automated scripts for experiment replication:

### Main Experimental Pipeline (`scripts/cj_test.sh`)
- **Baseline Experiments**: Standard evaluation protocols
- **MEC+BPC Experiments**: Multi-expert consensus with position balancing
- **Cascaded Evaluation**: Reliability-weighted judgment combination using Calibrated Judge
- **Model Configurations**: Support for multiple evaluation models

### Experimental Variants
- **Temperature Testing**: Sensitivity analysis across temperature settings
- **Cross-Validation**: Robust parameter estimation for Calibrated Scorer
- **Multi-Run Analysis**: Statistical significance assessment

### Running Experiments
```bash
# Set up environment
export CUDA_VISIBLE_DEVICES=0

# Run main experimental pipeline
bash scripts/cj_test.sh

# Run specific experiment variants
bash scripts/cj_temperature_test.sh
bash scripts/cj_cs_test_final.sh
```

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd thesis_code
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and model paths
   ```

4. **Download required models**:
   - Place local models in the `models/` directory
   - Ensure VLLM compatibility for reliability scoring

5. **Hardware Requirements**:
   - **GPU**: Minimum 50GB VRAM required for local model inference
   - **RAM**: Sufficient system memory for large model loading
   - **Storage**: Adequate space for model weights and datasets

## Dependencies

- **Core ML**: PyTorch, Transformers, VLLM
- **Evaluation**: OpenAI API, scikit-learn
- **Data Processing**: NumPy, Pandas
- **Utilities**: tqdm, python-dotenv