import copy
import gc
import json
import os
import random

import torch
import vllm
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from build_dataset import build_dataset
from build_prompt_judge import create_prompt, parse_predictions
from evaluate_judge import build_params


@torch.inference_mode()
def get_multi_answer(
        model_path,
        prompts,
        max_new_token=2048,
        temperature=0.1,
        top_p=1.0,
):
    """
    Generate multiple answers using a VLLM-compatible language model.

    Parameters:
    - model_path (str): Path to the local language model to load.
    - prompts (List[str]): A list of formatted input prompts.
    - max_new_token (int): Maximum number of tokens to generate in the output.
    - temperature (float): Sampling temperature for randomness.
    - top_p (float): Nucleus sampling parameter.

    Returns:
    - output_tokens (List[str]): List of generated responses (text format).
    - prefix_lens (List[int]): Lengths of input prompts (for masking).
    - target_lens (List[int]): Lengths of the model-generated responses.
    - output_ids (List[List[int]]): Full token IDs of prompt + generated output.
    """
    print("Start load VLLM model!")
    model = vllm.LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), dtype="bfloat16",
                     gpu_memory_utilization=0.8)
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_new_token,
        top_p=top_p,
    )
    print("VLLM model loaded!")

    tokenizer = model.get_tokenizer()
    MAX_LEN = model.llm_engine.model_config.max_model_len - 512  # Reserve space for generation
    prompt_ids = [tokenizer.encode(prompt)[-MAX_LEN:] for prompt in prompts]

    # Generate model outputs
    pred_list = model.generate(prompt_token_ids=prompt_ids, sampling_params=sampling_params)

    # Extract token IDs
    prompt_token_ids = [it.prompt_token_ids for it in pred_list]
    output_token_ids = [it.outputs[0].token_ids for it in pred_list]

    # Calculate lengths
    prefix_lens = [len(prompt_ids) for prompt_ids in prompt_token_ids]
    target_lens = [len(output_ids) for output_ids in output_token_ids]

    output_tokens = [it.outputs[0].text for it in pred_list]
    output_ids = [ids[0] + ids[1] for ids in zip(prompt_token_ids, output_token_ids)]

    return output_tokens, prefix_lens, target_lens, output_ids


@torch.inference_mode()
def get_single_evaluation(
        model,
        output_ids_ori,
        prefix_len,
        target_len,
):
    """
    Evaluate the model output for a single sample by computing:
    - Logit score: likelihood of generated tokens.
    - Entropy: uncertainty in predictions (positive correlation to calibration).
    - Variance: distribution spread of predicted token probabilities.

    Parameters:
    - model: Loaded transformer model.
    - output_ids_ori (Tensor): Token IDs including both prompt and generated output.
    - prefix_len (int): Length of the input prompt (for masking).
    - target_len (int): Length of the generated response (target of evaluation).

    Returns:
    - dict: Dictionary containing logit, entropy, and variance scores.
    """
    assert output_ids_ori.size()[0] == 1  # Batch size should be 1
    output_ids_ori = output_ids_ori.to(model.device)

    input_ids = copy.deepcopy(output_ids_ori)
    output_ids = output_ids_ori.clone()
    output_ids[0][:prefix_len] = -100  # Mask instruction tokens for loss

    # Forward pass to get logits
    outputs = model(
        input_ids=torch.as_tensor(input_ids),
        labels=output_ids,
        output_hidden_states=True,
        output_attentions=True,
    )

    # Shift token IDs for alignment
    shifted_input_ids = torch.roll(input_ids, shifts=-1)
    logprobs = torch.nn.functional.log_softmax(outputs["logits"], dim=-1)

    # Variance of logprobs over vocabulary
    logprobs_variance = torch.var(logprobs, dim=-1)
    logprobs_variance[output_ids == -100] = 0
    evaluation_var = logprobs_variance.sum(-1)[0] / target_len

    # Entropy (averaged over output tokens)
    logprobs[output_ids == -100] = 0
    logprobs_entropy = torch.mean(logprobs * outputs["logits"], dim=-1)
    evaluation_ent = logprobs_entropy.sum(-1)[0] / target_len

    # Logit score (mean log probability of generated tokens)
    evaluation_logit = torch.gather(logprobs, dim=-1, index=shifted_input_ids.unsqueeze(-1)).squeeze(-1)
    evaluation_logit = evaluation_logit.sum(-1)[0] / target_len

    return {"logit": evaluation_logit, "entropy": evaluation_ent, "variance": evaluation_var}


if __name__ == "__main__":
    parser = build_params()
    parser.add_argument("--cali-model-name-or-path", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = build_dataset(args.data_type)
    print(f"Loaded dataset from {args.data_path}")
    print(f"The length is {len(dataset)}")

    instruction = create_prompt(args.model_type, args.data_type)

    # Construct prompts only for compatible model_type and data_type
    prompts = []
    for index, example in enumerate(dataset):
        if args.model_type in ["judgelm", "auto-j"] and args.data_type in ["verbosity", "vicuna", "vicuna-mec",
                                                                           "vicuna-gpt4", "vicuna-mec-gpt4"]:
            example["rubric"] = "Please rate the helpfulness, relevance, accuracy, level of details of their responses."
            prompt = instruction.format(
                question_body=example["question_body"],
                rubric=example["rubric"],
                answer1_body=example["answer1_body"],
                answer2_body=example["answer2_body"]
            )
            prompts.append(prompt)

    print("Prompt construction finished. Sample prompt:")
    print(prompts[random.randint(0, len(prompts) - 1)] + "\n")

    # Save prompts to file
    with open('./prompts.txt', 'w') as f:
        for item in prompts:
            f.write(f"{item}\n")

    # Generate predictions
    predictions, prefix_lens, target_lens, output_ids = get_multi_answer(args.model_name_or_path, prompts,
                                                                         args.max_new_token)

    # Save generated predictions
    with open('./predictions.txt', 'w') as f:
        for item in predictions:
            f.write(f"{item}\n")

    gc.collect()
    torch.cuda.empty_cache()

    # Parse model predictions to structured format
    pred_scores = parse_predictions(predictions, args.model_type, args.data_type, args.prompt_type)

    # Save prediction scores
    with open(args.logit_file, "w", encoding="utf-8") as fout:
        for pred in pred_scores:
            fout.write(json.dumps(pred) + "\n")

    # Evaluate and record entropy for original model
    results = {"Entropy": []}

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).half().to(device)
    model.eval()

    for i in tqdm(range(len(predictions)), desc="Calculating reliability score"):
        evaluation = get_single_evaluation(
            model,
            torch.as_tensor([output_ids[i]]),
            prefix_lens[i],
            target_lens[i],
        )
        entropy = evaluation["entropy"]
        results["Entropy"].append(entropy.item() if isinstance(entropy, torch.Tensor) else entropy)
        gc.collect()
        torch.cuda.empty_cache()

    # Evaluate calibration model if provided
    if args.cali_model_name_or_path is not None:
        results["entropy_cali"] = []
        results["variance_cali"] = []
        model = AutoModelForCausalLM.from_pretrained(args.cali_model_name_or_path).half().to(device)
        model.eval()

        for i in tqdm(range(len(predictions)), desc="Calculating calibration reliability score"):
            evaluation = get_single_evaluation(
                model,
                torch.as_tensor([output_ids[i]]),
                prefix_lens[i],
                target_lens[i],
            )

            entropy = evaluation["entropy"]
            variance = evaluation["variance"]
            results["entropy_cali"].append(entropy.item() if isinstance(entropy, torch.Tensor) else entropy)
            results["variance_cali"].append(variance.item() if isinstance(variance, torch.Tensor) else variance)

    # Save all results
    with open(args.output_file, "w") as file_out:
        json.dump(results, file_out, indent=4)

    print(f"All reliability scores have been saved to {args.output_file}.")
