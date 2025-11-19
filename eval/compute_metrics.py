import json
import os
import glob
import numpy as np

from transformers import AutoTokenizer
from eval.custom_counter import count_frequencies_with_custom_equal
from eval.math_equivalence import is_equiv
from eval.math_util import my_answer_extraction

models = []
alpha_strategies = ["warmup"]
# alpha_strategies = ["constant", "warmup50", "warmup", "warmup200", "warmup500", "warmup1000"]
# alpha_strategies = ["warmup", "constant0.5", "constant0.75", "constant", "constant1.25", "constant1.5"] 

MAX_TOKENS = 8192
NUM_PREDS = 8
FIRST_N_QUESTIONS = -1
DATASETS = ["aime2025", "amc23"]
# DATASETS = ["aime2024", "aime2025", "amc23", "MATHL5", "gpqa"]
# NUM_PREDS = 1
# DATASETS = ["MATH_hard_test"]

small_size="1.5"
small_base_model=f"Qwen_Qwen2.5-Math-{small_size}B"
small_expert_model=f"deepseek-ai_DeepSeek-R1-Distill-Qwen-{small_size}B"
small_expert_model_llama=f"deepseek-ai_DeepSeek-R1-Distill-Llama-8B"
small_rl_expert_model=f"agentica-org_DeepScaleR-{small_size}B-Preview"
small_oneshot_rl_expert_model=f"ypwang61_One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1_pi13"
# large_size="7"
# large_base_model=f"Qwen_Qwen2.5-Math-{large_size}B"
large_size="32"
large_base_model=f"Qwen_Qwen2.5-{large_size}B"
large_base_instruct_model=f"Qwen_Qwen2.5-{large_size}B-Instruct"
large_expert_model=f"deepseek-ai_DeepSeek-R1-Distill-Qwen-{large_size}B"
large_ft_model=f"simplescaling_s1.1-{large_size}B"
large_base_llama_model="meta-llama_Llama-3.3-70B-Instruct"

"""
# Ablation for ThinkLogit-DPO
models += [f"dexperts-dpo-small_dpo_small_correct_small_incorrect-S{small_size}B-L{large_size}B/{alpha_strategy}" for alpha_strategy in alpha_strategies]
models += [f"dexperts-dpo-small_dpo_small_correct_large_incorrect-S{small_size}B-L{large_size}B/{alpha_strategy}" for alpha_strategy in alpha_strategies]
models += [f"dexperts-dpo-small_dpo_large_correct_small_incorrect-S{small_size}B-L{large_size}B/{alpha_strategy}" for alpha_strategy in alpha_strategies]
models += [f"dexperts-small_sft_large_correct-S{small_size}B-L{large_size}B/{alpha_strategy}" for alpha_strategy in alpha_strategies]
models += [f"dexperts-small_sft_small_correct-S{small_size}B-L{large_size}B/{alpha_strategy}" for alpha_strategy in alpha_strategies]
models += ["checkpoints_large_sft_on_small_correct"]
"""

# models += [f"dexperts-distill-lora-S{small_size}B-L{large_size}B/{alpha_strategy}" for alpha_strategy in alpha_strategies]


# models += ["simplescaling_s1.1-32B", "simplescaling_s1.1-32B/budget_forcing"]
# models += [small_expert_model, small_oneshot_rl_expert_model, large_ft_model, large_base_model, large_expert_model]
# models += [f"OneShotRLproxy-S{small_size}B-L{large_size}B/{alpha_strategy}" for alpha_strategy in alpha_strategies]
# models += [f"dexperts-dpo-lora64-rlvr-S{small_size}B-L{large_size}B/{alpha_strategy}" for alpha_strategy in alpha_strategies]
# models += [f"dexperts-dpo-lora-{sz}k-S{small_size}B-L{large_size}B/warmup" for sz in [5, 20, 30, 40, 50]]
# models += [f"dexperts-S{small_size}B-L{large_size}B/{alpha_strategy}" for alpha_strategy in alpha_strategies]
# models += [f"dexperts-dpo-lora-S{small_size}B-L{large_size}B/{alpha_strategy}" for alpha_strategy in alpha_strategies]
# models += [f"RLproxy-S{small_size}B-L{large_size}B/{alpha_strategy}" for alpha_strategy in alpha_strategies]

models += [
    # large_base_llama_model,
    # small_expert_model,
    # "qwen-guide-llama/warmup",
    # "qwen-guide-llama-dpo-transfer/warmup",
    # "qwen-guide-llama-dpo/warmup",
    # small_expert_model_llama,
    # Large model baselines
    # large_base_model,
    # large_ft_model,
    # large_expert_model,
    # large_base_instruct_model,
    f"{large_base_instruct_model}/budget_forcing",
    # SFT as guider
    # small_expert_model,
    # f"dexperts-S{small_size}B-L{large_size}B/warmup",
    # f"dexperts-dpo-lora-S{small_size}B-L{large_size}B/warmup",
    # RFT as guider
    # small_oneshot_rl_expert_model,
    # f"OneShotRLproxy-S{small_size}B-L{large_size}B/warmup",
    # f"dexperts-dpo-lora64-rlvr-S{small_size}B-L{large_size}B/warmup",
]


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B")

def average_token_count(texts):
    """
    Loads a Hugging Face tokenizer and calculates the average number of tokens
    for a list of input strings.

    Args:
        texts (list of str): The input texts to tokenize.
        model_name (str): The Hugging Face model name for the tokenizer.

    Returns:
        float: Average number of tokens per input string.
    """
    total_tokens = 0

    for text in texts:
        tokens = tokenizer.tokenize(text)
        total_tokens += len(tokens)

    avg_tokens = total_tokens / len(texts) if texts else 0
    return avg_tokens

def truncate_string_by_tokens(text, max_tokens=None):
    # Tokenize the input text
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    
    # Truncate to first N tokens
    truncated_ids = input_ids[:max_tokens]
    
    # Decode back to string
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    if "Human" in truncated_text:
        truncated_text = truncated_text.split("Human")[0]
    return truncated_text

def get_data_items(data_file):
    with open(data_file, 'r') as reader:
        data_items = [json.loads(l) for l in reader]
    for item in data_items:
        item["answer"] = str(item["answer"])
        item["preds"] = list()
    return data_items


def pass_at_k(data_items, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    
    n = len(data_items[0]['preds'])
    pass_at_k_items = []
    for item in data_items:
        assert n == len(item['preds'])
        c = sum([int(is_equiv(pred, item['answer'])) for pred in item['preds']])
        pass_at_k_items.append(estimator(n, c, k))
    return sum(pass_at_k_items) / len(pass_at_k_items)


def majority(data_items):
    maj_correct_scores = list()
    for item in data_items:
        pred_answer_frequencies = count_frequencies_with_custom_equal(item['preds'], is_equiv)
        try:
            voted_answer = pred_answer_frequencies[0][0]
        except Exception as e:
            print(item)
            exit(0)
        maj_correct_score = int(is_equiv(voted_answer, item["answer"]))
        maj_correct_scores.append(maj_correct_score)
    return sum(maj_correct_scores) / len(maj_correct_scores)

def main():
    for dataset in DATASETS:
        data_file = f"data/eval/{dataset}/test.jsonl"
        problem_type = "mutiple_choice" if "gpqa" in dataset.lower() else "math"

        for model in models:
            data_items = get_data_items(data_file)[:FIRST_N_QUESTIONS]

            pattern = f"results/{dataset}/{model}/*/predictions.jsonl"
            texts = []
            for file_path in glob.glob(pattern):
                with open(file_path, 'r') as reader:
                    pred_items = [json.loads(l) for l in reader]
                if len(pred_items) < len(data_items):
                    continue
                pred_items = pred_items[:len(data_items)]
                if len(data_items[0]["preds"]) >= NUM_PREDS:
                    break
                for pred_item, data_item in zip(pred_items, data_items):
                    model_output_truncated = truncate_string_by_tokens(pred_item["model_output"], max_tokens=MAX_TOKENS)
                    model_pred = my_answer_extraction(model_output_truncated, problem_type=problem_type)  
                    data_item["preds"].append(model_pred)
                    texts.append(model_output_truncated)
            if glob.glob(pattern): 
                k = len(data_items[0]["preds"])
                if k == 0:
                    continue
                majority_at_k = majority(data_items)
                avg_tokens = average_token_count(texts)
                print("="*50)
                print(f"{dataset}\t{model}\tmaj@{k} = {majority_at_k*100:.1f}\tavg_tokens = {avg_tokens:.0f}")
                for K_PASS in [1, NUM_PREDS]: 
                # for K_PASS in range(1, 2): 
                    pass_at_1 = pass_at_k(data_items, k=K_PASS)
                    print(f"{dataset}\t{model}\tpass@{K_PASS} = {pass_at_1*100:.1f}")



if __name__ == "__main__":
    main()
