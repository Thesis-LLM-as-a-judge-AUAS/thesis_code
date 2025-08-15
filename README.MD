# MultiEval framework

This repository contains the implementation and approbation of the approach introduced in the master project **"A Composite Methodology for Reducing Position and Length Biases in Cost-Effective LLM Evaluations"**.

![Architecture]("./figures/architecture.png")

## Purpose

The framework implements a novel approach that effectively mitigates position and length biases in large language model (LLM) evaluations while maintaining cost efficiency. It addresses two critical challenges in LLM evaluation:

- **Position Bias**: The tendency of evaluators to favor responses based on their presentation order
- **Length Bias**: The preference for longer responses regardless of actual quality

Through a composite methodology combining multiple evaluation signals, the framework provides robust, unbiased assessment of LLM performance at reduced computational costs.

## Codebase

The core evaluation components are located in the `code_components/` directory:

### Core Evaluation Modules

- **`fair_eval.py`**: Implements fair evaluation protocols with MEC+BPC methodology
  - Supports multiple evaluation models (GPT-3.5, GPT-4)
  - Implements k-sampling and position balancing
  - Implements core of Calibrated Judge

- **`cascaded_eval.py`**: Applies the cascaded evaluation methodology
  - Combines reliability scores with human judgments for optimal assessment
  - Implements intelligent routing based on reliability rankings
  - Handles forward/reverse order averaging for bias reduction
  - Merges multiple evaluation signals for robust outcomes
  - Implements a part of optimized judge tecnique

- **`cal_reliability.py`**: Generates responses from fine-tuned models and calculates reliability scores
  - Uses VLLM for efficient generation of multiple responses
  - Computes uncertainty-based reliability metrics (entropy, variance, logits)
  - Supports both auto-j and judgelm evaluation types
  - Generates reliability scores for cascaded evaluation
  - Implements a part of optimized judge tecnique

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

The framework supports multiple datasets and evaluation protocols designed for comprehensive bias mitigation research, with a focus on Calibrated Judge (CJ) and Calibrated Scorer (CS) methodology validation:

### Alpaca Datasets
Located in `datasets/alpaca/`, containing comprehensive evaluations for various language models:

#### Framework Datasets (`framework_datasets/`)
- **Base Models**: alpaca-7b, claude-2.1, gpt-3.5-turbo-1106, gpt4_0613, gpt4_1106_preview, Mixtral-8x7B-Instruct-v0.1
- **Verbosity Variants**: Each model has three versions:
  - **Standard**: Original model responses
  - **Concise**: Shortened responses for length bias analysis
  - **Verbose**: Extended responses for verbosity bias analysis
- **Evaluation Types**: 
  - Standard AlpacaEval with GPT-4 annotations
  - Chain-of-Thought (CoT) evaluation with GPT-4 Turbo
  - Weighted evaluation protocols

#### CJ-CS Test Datasets (`cj-cs-test/`)
- **Auto-J Evaluation**: `alpaca-auto-j.json` and `alpaca-auto-j-gpt4.json`
- **JudgeLM Evaluation**: `alpaca-judgelm.json` and `alpaca-judgelm-gpt4.json`
- **MEC+BPC Experiments**: Position-balanced question-answer pairs
- **Balanced Dataset**: `balanced.json` for last experiments
- **Expanded Judgments**: Prepared dataset for verbosity experiment

### Vicuna Datasets
Located in `datasets/vicuna/`, featuring instruction-following evaluation:

#### Core Dataset
- **Questions**: `question.jsonl` - 80 diverse instruction-following tasks
- **Categories**: Generic, knowledge, roleplay, common-sense, fermi estimation, counterfactual, coding, math, writing
- **Ground Truth**: Human-annotated preference labels

#### Model Responses
- **Multiple Models**: Vicuna-13B, GPT-3.5, GPT-4, Alpaca
- **Response Format**: JSONL with metadata including generation parameters
- **Evaluation Metadata**: HuggingFace arguments, temperature settings, token limits

#### Sampled Data (`sampled_data/`)
- **CJ Sampled**: Calibrated Judge evaluation subsets
  - **Questions**: 10-question subsets (0-9) for experimental efficiency
  - **Model Answers**: Paired responses for comparison evaluation
  - **Merged Versions**: Combined datasets for comprehensive analysis
- **Temperature Investigation**: Subsets for temperature sensitivity analysis of CJ core

### Experimental Data
Located in `gathered_data/`, containing comprehensive experimental outcomes:

#### Baseline and MEC Results
- **Baseline Experiments**: Standard evaluation protocols
- **MEC+BPC Results**: Multi-Expert Consensus with Position Balancing outcomes
- **40 Experimental Runs**: Statistical significance through repeated experiments

#### Cascaded Evaluation Results
- **Reliability Scores**: Model uncertainty and calibration metrics
- **Final Results**: Combined reliability and judgment outcomes
- **Different Presets**: vicuna, vicuna-mec, vicuna-gpt4, vicuna-mec-gpt4

### CJ+CS Combination Test Results

#### Temperature Investigation of CJ Core
- **Temperature Sensitivity**: Analysis of CJ reliability across different temperature settings
- **Optimal Parameters**: Identification of temperature ranges for maximum reliability
- **Stability Analysis**: Assessment of CJ performance consistency across parameter variations

#### CJ+CS Integration Performance
- **Combined Methodology**: Results from integrating Calibrated Judge with Calibrated Scorer
- **Bias Mitigation Effectiveness**: Quantitative assessment of position and length bias reduction
- **Performance Comparison**: CJ+CS vs. CJ

## Repository structure

```
thesis_code/
├── analysis/                 # Notebooks with results analysis and data sampling
├── code_components/          # Python files implementing Calibrated Judge and Calibrated Scorer
├── datasets/                 # Vicuna and alpaca datasets + sampled/changed versions
├── figures/                  # Pictures from the paper
├── gathered_data/            # Data gathered during experiments
├── models/                   # Models necessary to download for CJ experiments (gitignored)
├── scripts/                  # Start scripts for gathering data for experiments
└── requirements.txt          # Python dependencies
```

## Preliminaries for Experiment Replication

This section provides step-by-step instructions for setting up the environment to replicate the experiments in this framework.

### Hardware Setup

#### GPU Requirements
- **Minimum VRAM**: 50GB GPU memory required
- **Recommended**: NVIDIA A100 (80GB) or H100 (80GB) for optimal performance
- **Alternative**: Multiple GPUs with combined VRAM ≥ 50GB
- **Compatibility**: CUDA 11.8+ support required

#### System Requirements
- **RAM**: Minimum 32GB system memory
- **Storage**: At least 200GB free space for models and datasets
- **CPU**: Multi-core processor for data preprocessing

### Software Installation

#### 1. Operating System Setup
```bash
# Ubuntu 20.04+ or CentOS 8+ recommended
# Ensure system is up to date
sudo apt update && sudo apt upgrade -y  # Ubuntu/Debian
# or
sudo yum update -y  # CentOS/RHEL
```

#### 2. Python Installation
```bash
# Install Python 3.9+ (required for VLLM compatibility)
sudo apt install python3.9 python3.9-venv python3.9-dev python3-pip -y

# Verify installation
python3.9 --version
pip3 --version
```

#### 3. Bash Installation
```bash
# Install/update bash (usually pre-installed on Linux)
sudo apt install bash -y

# Verify version (4.4+ recommended)
bash --version
```

#### 4. CUDA and GPU Drivers
```bash
# Install NVIDIA drivers (if not already installed)
sudo apt install nvidia-driver-535 -y

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
nvidia-smi
```

### Environment Setup

#### 1. Create Virtual Environment
```bash
# Navigate to project directory
cd thesis_code

# Create virtual environment
python3.9 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation
which python
python --version
```

#### 2. Install Python Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify key packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import vllm; print(f'VLLM: {vllm.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

#### 3. Install Jupyter Notebook
```bash
# Install Jupyter
pip install jupyter notebook

# Install additional analysis packages
pip install matplotlib seaborn plotly

# Verify installation
jupyter --version
```

### Model Installation

#### 1. Create Models Directory
```bash
# Create models directory
mkdir -p models
cd models
```

#### 2. Download Required Models

##### JudgeLM-7B-v1.0 (Calibrated Judge)
```bash
# Download JudgeLM model
git lfs install
git clone https://huggingface.co/BAAI/JudgeLM-7B-v1.0
cd JudgeLM-7B-v1.0
# Model size: ~13.5GB
cd ..
```

##### Vicuna-7B-v1.5 (Base Model)
```bash
# Download Vicuna model
git clone https://huggingface.co/lmsys/vicuna-7b-v1.5
cd vicuna-7b-v1.5
# Model size: ~13.5GB
cd ..
```

##### Mistral-7B-v0.1 (Base Model)
```bash
# Download Mistral model
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1
cd Mistral-7B-v0.1
# Model size: ~13.5GB
cd ..
```

##### AutoJ-13B (Calibrated Judge)
```bash
# Download AutoJ model
git clone https://huggingface.co/GAIR/autoj-13b
cd autoj-13b
# Model size: ~26GB
cd ..
```

#### 3. Verify Model Installation
```bash
# Check model directories
ls -la models/

# Verify model files
du -sh models/*/

# Test model loading (optional - requires significant VRAM)
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Test loading a smaller model first
model_name = 'models/Mistral-7B-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')
print('Model loaded successfully!')
"
```

### Environment Configuration

#### 1. Update .env File
```bash
# Navigate back to project root
cd ..

# Create/update .env file
cat > .env << EOF
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
EOF
```

#### 2. Verify Setup
```bash
# Test basic functionality
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('Environment variables loaded successfully!')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```
## Experiments Replication

This section provides the complete workflow for replicating all experiments in the MultiEval framework. Follow the steps in order to ensure proper data flow and result generation.

### 1. Data Sampling and Preparation

Start by running the data sampling notebook to prepare the experimental datasets:

```bash
# Activate virtual environment
source venv/bin/activate

# Launch Jupyter Notebook
jupyter notebook

# Navigate to analysis/dataset_sampling.ipynb
# Run all cells to generate sampled datasets
```

### 2. Script Execution (If Required by Chapter)

Some chapters in the analysis notebook may require running specific scripts. Check the notebook instructions and execute any required scripts:

```bash
# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Run specific scripts as directed by the notebook
bash scripts/cj_cs_test_cal_reliability.sh
bash scripts/cj_cs_test_cascaded_eval.sh
```

### 3. Execute All Experimental Scripts

Run all experimental scripts to gather comprehensive data:

```bash
bash scripts/cj_test.sh
bash scripts/cj_temperature_test.sh
bash scripts/cj_cs_test.sh
```

**Expected Output**: 
- Experimental results in `gathered_data/` directory

### 4. Results Analysis and Visualization

Finally, run the comprehensive results analysis notebook:

```bash
# Launch Jupyter Notebook
jupyter notebook

# Navigate to analysis/experimental_result_analysis.ipynb
# Run all cells to generate complete analysis
```

**Purpose**: This notebook contains all the results presented in the report, providing comprehensive analysis of the experimental outcomes.