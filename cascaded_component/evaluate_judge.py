import argparse

def build_params():
    """
    Creates and returns an ArgumentParser instance configured with CLI arguments
    for a script that likely deals with evaluating or running language models.

    The arguments include:
    - --model-name-or-path: Path or name of the pretrained model to load.
    - --prompt-type: Type of prompting strategy used. Supported options:
        * "vanilla": standard prompting
        * "cot": chain-of-thought prompting
        * "icl": in-context learning
    - --model-type: Specifies the model evaluation type, e.g., "judgelm" or "auto-j".
    - --data-type: Type of evaluation dataset (e.g., "verbosity", "vicuna", etc.).
    - --data-path: Directory where input data is located. Defaults to "./data".
    - --max-new-token: Maximum number of tokens the model can generate. Default is 2048.
    - --temperature: Sampling temperature for generation. Lower values make output more deterministic.
    - --top-p: Nucleus sampling parameter. Controls diversity of output by limiting token selection to top cumulative probability p.
    - --logit-file: Optional path to a file where logits or model outputs can be saved or loaded.

    Returns:
        argparse.ArgumentParser: Configured parser with all necessary arguments defined.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=("judgelm", "auto-j"),
        default=None,
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=("verbosity", "vicuna", "vicuna-mec", "vicuna-gpt4", "vicuna-mec-gpt4"),
        default=None,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=2048,
        help="The maximum number of new tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--logit-file",
        type=str,
        default=None
    )
    return parser
