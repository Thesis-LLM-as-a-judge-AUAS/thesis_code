import json
import random
import numpy as np

from evaluate_judge import build_params


def load_results(file_path):
    """
    Load scalar metrics (e.g., reliability scores) from a JSON file.

    Expected JSON structure example:
    {
        "Entropy": [0.12, 0.34, ...],
        "SomeOtherMetric": [...]
    }

    Args:
        file_path (str): Path to a JSON file containing per-item scores.

    Returns:
        dict: Parsed JSON dictionary with metric names as keys and lists of scores as values.
    """
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results


def get_average_scores(scores):
    """
    For datasets where each item appears twice (forward and reverse order),
    compute a single score per original item by averaging the forward and reverse scores.

    Assumptions:
      - The input 'scores' list contains 2N elements.
      - The first N scores correspond to forward presentation.
      - The last N scores correspond to the same items in reverse presentation.
      - Items are aligned by index across the two halves.

    Args:
        scores (list[float]): Sequence of 2N scores (forward + reverse).

    Returns:
        list[float]: Length-N list with the mean of forward[i] and backward[i].
    """
    mid_index = len(scores) // 2
    forward_scores = scores[:mid_index]
    backward_scores = scores[mid_index:]
    average_scores = [(forward_scores[i] + backward_scores[i]) / 2 for i in range(mid_index)]
    return average_scores


def select_top_half_indices(average_scores, total_length):
    """
    Select the top-50% items by averaged score and include both their forward
    and reverse indices.

    This is useful when you need to keep both presentations of the same
    underlying item in downstream processing.

    Args:
        average_scores (list[float]): Averaged scores of length N.
        total_length (int): Total length of the original score list (= 2N).

    Returns:
        np.ndarray: Indices of shape (N,) containing both forward and reverse
                    positions for the top half by score.
    """
    sorted_indices = np.argsort(-np.array(average_scores))
    top_half_indices = sorted_indices[:len(sorted_indices) // 2]
    indices_with_reverse = np.concatenate([top_half_indices, top_half_indices + total_length // 2])
    return indices_with_reverse


def compute_accuracy_rate_weak_weak(relia_scores1, relia_scores2, judge_output1, judge_output2, answers, dataset_type):
    """
    Route each item’s final judgment to the 'weaker' candidate whose reliability rank is better,
    using two parallel reliability signals (relia_scores1, relia_scores2).

    Logic:
      1) Convert reliability scores to ranks (lower rank = more reliable for that item).
      2) For each item, select the output from the source with the better (lower) rank.
      3) If ranks tie, break ties randomly.
      4) (Intended) Compare the selected output to the ground-truth to compute accuracy.

    Notes:
      - If dataset_type == "auto-j", we first average forward/reverse halves
        via get_average_scores(...).

    Args:
        relia_scores1 (list[float]): Reliability scores for system 1.
        relia_scores2 (list[float]): Reliability scores for system 2.
        judge_output1 (list[Any]): Per-item judgments from system 1.
        judge_output2 (list[Any]): Per-item judgments from system 2.
        answers (list[Any]): Ground-truth labels/answers per item.
        dataset_type (str): Dataset flag; "auto-j" triggers forward/reverse averaging.

    Returns:
        None
        (Side-effect only in this snippet; you may extend it to return accuracy, etc.)
    """

    def get_sort_indices(relia_scores):
        """
        Convert raw scores to per-item ranks.
        Higher score -> earlier in sorted order -> lower numeric rank.
        """
        sorted_indices = list(np.argsort(-np.array(relia_scores)))
        # For each original index i, find its position in the descending sort -> rank
        ranks = [sorted_indices.index(i) for i in range(len(sorted_indices))]
        return ranks

    if dataset_type == "auto-j":
        relia_scores1 = get_average_scores(relia_scores1)
        relia_scores2 = get_average_scores(relia_scores2)

    sorted_indices1 = get_sort_indices(relia_scores1)  # ranks for system 1
    sorted_indices2 = get_sort_indices(relia_scores2)  # ranks for system 2

    judge_outputs = []
    judge_answers = []

    for rank1, rank2, output1, output2, answer in zip(
        sorted_indices1, sorted_indices2, judge_output1, judge_output2, answers
    ):
        if rank1 < rank2:
            # System 1 is more reliable (better rank) for this item
            judge_outputs.append(output1)
            judge_answers.append(answer)
        elif rank2 < rank1:
            # System 2 is more reliable (better rank) for this item
            judge_outputs.append(output2)
            judge_answers.append(answer)
        else:
            # Tie: randomly choose one system’s output
            judge_outputs.append(random.choice([output1, output2]))
            judge_answers.append(answer)

    # You can compute accuracy like:
    # accuracy = np.mean([pred == gt for pred, gt in zip(judge_outputs, judge_answers)])
    # return accuracy


def compute_accuracy_rate_weak_strong(
    relia_scores1,
    judge_output1,
    judge_output_gpt,
    dataset_type,
    final_output_path,
    ratio
):
    """
    Hybrid routing between a weaker and a stronger judge based on per-item reliability
    from the weaker judge.

    Strategy:
      - Rank items by the weaker judge’s reliability (descending).
      - Take the top 'ratio' fraction of items and keep the weaker judge’s outputs.
      - Route the remaining items to the stronger judge (e.g., GPT-4) to improve quality.
      - Optionally, for datasets like 'auto-j' (forward/reverse duplicates), expand the
        selected indices to include both halves.

    Args:
        relia_scores1 (list[float]): Reliability scores from the weaker judge (length N or 2N for auto-j).
        judge_output1 (list[Any]): Per-item outputs from the weaker judge.
        judge_output_gpt (list[Any]): Per-item outputs from the stronger judge.
        dataset_type (str): If "auto-j", treat scores as paired halves (forward/reverse).
        final_output_path (str): Where to write routed outputs as JSON.
        ratio (float): Fraction of items to keep with the weaker judge (e.g., 0.5 -> top 50%).

    Returns:
        None (writes routed outputs to 'final_output_path').
    """

    def get_top_half_indices(relia_scores, dataset_type, ratio=0.9):
        """
        Rank items by reliability and return indices for the top 'ratio' fraction.
        For 'auto-j', also include paired reverse indices to keep both presentations aligned.
        """
        sorted_indices = np.argsort(-np.array(relia_scores))
        top_half_indices = sorted_indices[:int(len(sorted_indices) * ratio)]

        if dataset_type == "auto-j":
            # Mirror indices to include the reverse half (assumes 2N layout)
            top_half_indices = np.concatenate(
                [top_half_indices, top_half_indices + len(sorted_indices)]
            )

        return list(top_half_indices)

    if dataset_type == "auto-j":
        relia_scores1 = get_average_scores(relia_scores1)

    top_half_indice1 = set(get_top_half_indices(relia_scores1, dataset_type, ratio))

    judge_outputs = []
    for i, output1, output2 in zip(np.arange(len(judge_output1)).tolist(), judge_output1, judge_output_gpt):
        # If the item is among the top reliable ones for the weak judge, keep its decision;
        # otherwise, defer to the strong judge’s output.
        if i in top_half_indice1:
            judge_outputs.append(output1)
        else:
            judge_outputs.append(output2)

    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(judge_outputs, f, ensure_ascii=False, indent=4)


def main():
    """
    Entry point:
      - Parse CLI args.
      - Load reliability scores and judge outputs.
      - Run weak-strong routing with a specified ratio.
      - Persist the routed outputs to disk.
    """
    random.seed(42)
    np.random.seed(42)

    parser = build_params()
    parser.add_argument("--output-file1", type=str, default=None, help="JSON file with reliability scores (e.g., contains key 'Entropy').")
    parser.add_argument("--logit-file1", type=str, default=None, help="Per-line JSON outputs from the weaker judge.")
    parser.add_argument("--output-file2", type=str, default=None, help="(Optional) Second reliability source for weak-weak mode.")
    parser.add_argument("--logit-file2", type=str, default=None, help="(Optional) Per-line JSON outputs from the second weak judge.")
    parser.add_argument("--logit-file-gpt", type=str, default=None, help="Per-line JSON outputs from the stronger judge, must contain 'score' field per line.")
    parser.add_argument("--final-output-file", type=str, default=None, help="Path to write routed outputs as a JSON list.")
    args = parser.parse_args()

    # Load reliability scores for the weak judge
    relia_scores1 = load_results(args.output_file1)["Entropy"]

    # Load weak judge outputs (per-line JSON objects)
    with open(args.logit_file1, 'r') as f:
        judge_output1 = [json.loads(line.strip()) for line in f.readlines()]

    # Load strong judge outputs (per-line JSON objects with 'score')
    with open(args.logit_file_gpt, 'r') as f:
        judge_output_gpt = [json.loads(line.strip())["score"] for line in f.readlines()]

    # Route items by reliability (keep top 'ratio' with the weak judge, defer the rest to the strong judge)
    compute_accuracy_rate_weak_strong(
        relia_scores1=relia_scores1,
        judge_output1=judge_output1,
        judge_output_gpt=judge_output_gpt,
        dataset_type=args.data_type,
        final_output_path=args.final_output_file,
        ratio=0.5
    )


if __name__ == "__main__":
    main()