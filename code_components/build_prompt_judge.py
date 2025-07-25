import random

def create_prompt(model_type):
    """
    Generate an evaluation prompt template based on the model and data type.

    Parameters:
    - model_type (str): The type of model (e.g., 'judgelm', 'auto-j').

    Returns:
    - str: A formatted prompt template string with placeholders for question and answers.
    """
    if model_type == "judgelm":
        # Prompt format for evaluating two assistant responses with score and explanation
        instruction = """You are a helpful and precise assistant for checking the quality of the answer.
[Question]
{question_body}

[The Start of Assistant 1's Answer]
{answer1_body}

[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{answer2_body}

[The End of Assistant 2's Answer]

[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

### Response:"""

    elif model_type == "auto-j":
        # Prompt for Auto-J style evaluation with forced final decision between responses
        instruction = """[INST] You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
***
[Query]: {question_body}
***
[Response 1]: {answer1_body}
***
[Response 2]: {answer2_body}
***
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided. [/INST]"""

    return instruction

def parse_predictions(predictions, model_type):
    """
    Parse the predictions returned by the LLMs into numerical scores.

    Parameters:
    - predictions (List[str]): A list of raw prediction strings from the model.
    - model_type (str): The model that generated the responses ('judgelm', 'auto-j').

    Returns:
    - List: Parsed scores as either [score1, score2] for pairwise or a single float for individual scores.
    """
    def parse_score_judgelm(review):
        """Parse Judgelm-style pairwise score from response text."""
        try:
            score_pair = review.split('\n')[-1]
            score_pair = score_pair.replace(',', ' ').strip()
            sp = score_pair.split()
            return [float(sp[0]), float(sp[1])]
        except:
            return [1.0, 1.0]  # default to tie

    def parse_score_autoj(review):
        """Parse Auto-J style decision for which response is better or tied."""
        review = review.strip().lower()
        pos = review.rfind('final decision is ')
        if pos != -1:
            pred_rest = review[pos + len('final decision is '):].strip()
            if pred_rest.startswith('response 1'):
                return [1, 0]
            elif pred_rest.startswith('response 2'):
                return [0, 1]
            elif pred_rest.startswith('tie'):
                return [1, 1]
        return [1.0, 1.0]  # default to tie

    # Apply appropriate parser based on model type
    if model_type == "judgelm":
        pred_scores = [parse_score_judgelm(pred.strip()) for pred in predictions]
    elif model_type == "auto-j":
        pred_scores = [parse_score_autoj(pred) for pred in predictions]

    # Log sample for verification
    print("Prediction parsing finished!")
    print("Sampled prediction:")
    random_idx = random.randint(0, len(predictions) - 1)
    print(predictions[random_idx])
    print(f"Sampled score: {pred_scores[random_idx]}")

    return pred_scores
