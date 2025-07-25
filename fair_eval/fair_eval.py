import argparse
import json
import os
import time

import openai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv('.env')

# Constants
MAX_API_RETRY = 10000
REQ_TIME_GAP = 4

# Arguments from bash
parser = argparse.ArgumentParser()
parser.add_argument("-q", "--question-file")
parser.add_argument("-a", "--answer-file-list", nargs="+", default=[])
parser.add_argument('-o', '--output', help='Output file (defaults to stdout)')
parser.add_argument("-m", "--eval-model", default="gpt-3.5-turbo")
parser.add_argument("-k", "--k", type=int, default=3)
parser.add_argument("-b", "--bpc", type=int, default=1)

args = parser.parse_args()

# Cost calculation constants
if args.eval_model == "gpt-4":
    cost_per_promtp_token = 0.03 / 1000
    cost_per_completion_token = 0.06 / 1000
elif args.eval_model == "gpt-3.5-turbo":
    cost_per_promtp_token = 2 / 10 ** 6
    cost_per_completion_token = 2 / 10 ** 6
else:
    raise ValueError("Invalid evaluator name")

# OpenAI API key and caching
os.environ["OPENAI_ENABLE_CACHE"] = "true"

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# Judge prompt
def gen_prompt(question, first_answer, second_answer):
    sys_prompt = 'You are a helpful and precise assistant for checking the quality of the answer.'
    prompt_template = "[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n"
    default_prompt = """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
    Please rate the helpfulness, relevance, accuracy, level of details of their responses. 

    Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
    Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. 
    Then, output two lines indicating the scores for Assistant 1 and 2, respectively.

    Output with the following format:
    Evaluation evidence: <your evaluation explanation here>
    Score of the Assistant 1: <score>
    Score of the Assistant 2: <score>"""
    return sys_prompt, prompt_template.format(question=question, answer_1=first_answer, answer_2=second_answer,
                                              prompt=default_prompt)


# Query function
def query_gpt(system_prompt, uer_prompt):
    for retry_idx in range(MAX_API_RETRY):
        try:
            response = client.chat.completions.create(
                model=args.eval_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": uer_prompt},
                ],
                temperature=1,
                max_tokens=512,
                n=args.k
            )

            return response
        except openai.RateLimitError as e:
            print('Rate limit')
            print(e)
            time.sleep(30)
        except Exception as e:
            print(f'error: {e}')
    raise RuntimeError(f"Failed after {MAX_API_RETRY} retries.")


def process_and_calculate_cost_for_prompt(question, first_answer, second_answer):
    # Init final cost
    final_cost = 0

    # Prompt the judge
    system_prompt, user_prompt = gen_prompt(question, first_answer, second_answer)
    response = query_gpt(system_prompt, user_prompt)

    # Calculate the cost of prompting
    final_cost += response.usage.prompt_tokens * cost_per_promtp_token
    final_cost += response.usage.completion_tokens * cost_per_completion_token

    return response, final_cost


# Number of retries
N_RETRIES = 20


def extract_scores(question, first_answer, second_answer):
    for retry_idx in range(N_RETRIES):
        try:
            response, final_cost = process_and_calculate_cost_for_prompt(question, first_answer, second_answer)

            all_scores = []
            content_bodies = []

            for choice in response.choices:
                # Extract the score from judgement (if exist)
                content = choice.message.content
                first_score, second_score = parse_score_from_review(content)

                if first_score == -1 or second_score == -1:
                    continue

                # Save answer and scores
                all_scores.append([first_score, second_score])
                content_bodies.append(content)

            return all_scores, content_bodies, final_cost
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Failed after {N_RETRIES} retries")


def get_eval(question, first_answer, second_answer):
    # Get the first scores
    all_scores, content_bodies, final_cost = extract_scores(question, first_answer, second_answer)

    # Backup initialization
    contents_bpc = []

    # Flag for usage Balance Position Calibration
    if args.bpc == 1:
        # Extract scores again for BPC
        scores_bpc, contents_bpc, bpc_cost = extract_scores(question, second_answer, first_answer)

        final_cost += bpc_cost
        all_scores = all_scores + [sub[::-1] for sub in scores_bpc]

    # Calculate final average score
    first_score = sum([score[0] for score in all_scores]) / len(all_scores)
    second_score = sum([score[1] for score in all_scores]) / len(all_scores)

    return content_bodies, contents_bpc, final_cost, [first_score, second_score]


# Extract the score from the response of judge
def parse_score_from_review(review):
    # Extract the score from the row before the last + the number after :
    first_score = review.split("\n")[-2].split(":")[-1].strip()
    second_score = review.split("\n")[-1].split(":")[-1].strip()
    return [float(first_score), float(second_score)]


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list


# Gather all the judgements into one list
def extract_results(gathered_reviews):
    final_results = []

    for idx, (contents, contents_bpc, cost, [score1, score2]) in enumerate(gathered_reviews):
        final_results.append({
            "question_id": question_jsons[idx]["question_id"],
            "question": question_jsons[idx]["text"],
            "review": contents,
            "review_bpc": contents_bpc,
            "cost": cost,
            "score": [score1, score2],
        })

    return final_results


# Calculate final cost and ratio
def conclude_the_results(final_results):
    # Init the result vars
    total_cost = 0
    judgement_results = {
        'win': 0,
        'tie': 0,
        'loss': 0
    }

    for current_result in final_results:
        # Extract cost and scores
        first_score, second_score = current_result["score"]
        total_cost += current_result["cost"]

        # Find the result
        if first_score == second_score:
            judgement_results['tie'] += 1
        elif first_score > second_score:
            judgement_results['win'] += 1
        else:
            judgement_results['loss'] += 1

    return total_cost, judgement_results


# Save final judgements
def save_the_results(final_results):
    with open(f"{args.output}", "w") as output_review_file:
        for current_result in final_results:
            # Save result judgement
            output_review_file.write(json.dumps(current_result) + "\n")


if __name__ == "__main__":
    try:
        # Upload jsons with questions and answers
        question_jsons = get_json_list(args.question_file)
        first_answers_json = get_json_list(args.answer_file_list[0])
        second_answers_json = get_json_list(args.answer_file_list[1])

        # Check equality of length
        assert len(question_jsons) == len(first_answers_json) == len(second_answers_json)

        # Init vars
        reviews = []
        total_len = len(question_jsons)
        question_idx_list = list(range(total_len))

        # Iterate with progress bar through all the questions
        for i in tqdm(question_idx_list):
            # Check that in all datasets are the same question
            assert (
                    first_answers_json[i]["question_id"]
                    == question_jsons[i]["question_id"]
                    == second_answers_json[i]["question_id"]
            )

            question = question_jsons[i]["text"]
            first_answer = first_answers_json[i]["text"]
            second_answer = second_answers_json[i]["text"]

            # Extract the evaluation
            reviews.append(get_eval(question, first_answer, second_answer))

            # To avoid the rate limit set by OpenAI
            time.sleep(REQ_TIME_GAP)

        # Find the results
        final_results = extract_results(reviews)

        # Save results
        save_the_results(final_results)

        # Conclude the major discoveries
        total_cost, judgement_results = conclude_the_results(final_results)

        print(f'Evaluation results (The first model vs the second model):\n{judgement_results}')
        print(f'Evaluation cost: ${total_cost:.2f}.')

    finally:
        with open(f'./{time.time()}-reviews.json', 'w') as f:
            json.dump(reviews, f)
