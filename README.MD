# LLM Evaluation Framework

This repository contains a modular framework for evaluating large language models (LLMs) in alignment with human judgments, while mitigating evaluation biases and reducing computational costs.

## Structure

- `sanity_check.ipynb`  
  Main notebook containing the full experimental workflow, statistical tests, visualizations, and final conclusions.

- `datasets/`  
  Directory containing the original evaluation datasets (e.g., AlpacaEval, Vicuna) used as inputs for model judgment and scoring.

- `gathered_data/`  
  Stores all experimental outputs, including model responses, judgment annotations, and evaluation logs collected during runtime.

- `cascaded_component/`  
  Module implementing the cost reduction mechanism based on cascading decision logic to minimize reliance on expensive models.

- `lc_component/`  
  Contains the length calibration logic used to mitigate verbosity-related bias in model preference assessments.

- `FairEval.py`  
  Script implementing position bias mitigation using MEC + BPC, forming the core of the Calibrated Judge method.

## Overview

The repository supports evaluation across three dimensions: position bias, verbosity bias, and computational cost. It integrates and extends recent bias-aware techniques into a cohesive evaluation stack for reliable LLM benchmarking.
