DATASETS = ["aime2024", "aime2025", "amc23", "MATHL5", "gpqa", "lcb_200"]
# DATASETS = ["aime2024"]
NUM_PREDS = 8
MAX_TOKENS = 8192

import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
import re
from tqdm import tqdm
import spacy
import os

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

nlp = spacy.load("en_core_web_sm")

models = []
models += [
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B", # Guider
    "Qwen_Qwen2.5-32B", # Target
    "dexperts-S1.5B-L32B/warmup", # Taget + ThinkLogit
    "dexperts-dpo-lora-S1.5B-L32B/warmup", # Target + ThinkLogit-DPO
    # "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B" # Target + Full Finetune
]

branching_keywords = ["alternatively", "another", "try", "suppose", "consider", "different", "assume", "also", "option"]
backtracking_keywords = ["however", "but", "mistake", "error", "contradiction", "wrong", "revisit", "actually", "again", "flawed"]
self_verification_keywords = ["check", "verify", "confirm", "satisfy", "plug", "back", "substitute", "ensure", "validate", "test"]

behaviors = ['branching', 'backtracking', 'self_verification']
behavior_names = ['Branching out', 'Backtracking', 'Self-verification']

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
        if "answer" in item:
            item["answer"] = str(item["answer"])
        item["preds"] = list()
    return data_items

def main():
    counts_file = 'behaviour_analysis_counts.json'
    if os.path.exists(counts_file):
        with open(counts_file, 'r') as f:
            counts_data = json.load(f)
        counts_raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        counts_norm = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for ds in counts_data['raw']:
            for mod in counts_data['raw'][ds]:
                for beh in counts_data['raw'][ds][mod]:
                    counts_raw[ds][mod][beh] = counts_data['raw'][ds][mod][beh]
        for ds in counts_data['norm']:
            for mod in counts_data['norm'][ds]:
                for beh in counts_data['norm'][ds][mod]:
                    counts_norm[ds][mod][beh] = counts_data['norm'][ds][mod][beh]
    else:
        counts_raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        counts_norm = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for dataset in DATASETS:
        data_file = f"data/eval/{dataset}/test.jsonl"
        FIRST_N_QUESTIONS = 200 if dataset == "lcb_200" else None
        data_items = get_data_items(data_file)[:FIRST_N_QUESTIONS]

        for model in models:
            if dataset in counts_norm and model in counts_norm[dataset]:
                print(f"Skipping already processed {dataset} {model}")
                continue
            print(f"Processing model {model} on dataset {dataset}...")

            pattern = f"../GL_bak/proxy-tuning/results/{dataset}/{model}/*/predictions.jsonl"
            file_count = 0
            for file_path in glob.glob(pattern):
                if file_count >= NUM_PREDS:
                    break
                with open(file_path, 'r') as reader:
                    pred_items = [json.loads(l) for l in reader]
                if len(pred_items) < len(data_items):
                    continue
                pred_items = pred_items[:len(data_items)]
                file_count += 1
                for pred_item in tqdm(pred_items):
                    model_output_truncated = truncate_string_by_tokens(pred_item["model_output"], max_tokens=MAX_TOKENS)
                    text = model_output_truncated.lower()
                    doc = nlp(text)
                    lemmas = [token.lemma_ for token in doc]
                    length = len(lemmas)
                    branching_count = sum(lemmas.count(kw) for kw in branching_keywords)
                    backtracking_count = sum(lemmas.count(kw) for kw in backtracking_keywords)
                    self_verification_count = sum(lemmas.count(kw) for kw in self_verification_keywords)
                    counts_raw[dataset][model]['branching'].append(branching_count)
                    counts_raw[dataset][model]['backtracking'].append(backtracking_count)
                    counts_raw[dataset][model]['self_verification'].append(self_verification_count)
                    if length > 0:
                        branching_norm = branching_count / length
                        backtracking_norm = backtracking_norm = backtracking_count / length
                        self_verification_norm = self_verification_count / length
                    else:
                        branching_norm = backtracking_norm = self_verification_norm = 0
                    counts_norm[dataset][model]['branching'].append(branching_norm)
                    counts_norm[dataset][model]['backtracking'].append(backtracking_norm)
                    counts_norm[dataset][model]['self_verification'].append(self_verification_norm)
            assert file_count == NUM_PREDS, f"Expected {NUM_PREDS} prediction files for model {model} on dataset {dataset}, but found {file_count}."
            
            # Save counts after each dataset + model combination
            counts_data = {
                'raw': {ds: {mod: dict(counts_raw[ds][mod]) for mod in counts_raw[ds]} for ds in counts_raw},
                'norm': {ds: {mod: dict(counts_norm[ds][mod]) for mod in counts_norm[ds]} for ds in counts_norm}
            }
            with open(counts_file, 'w') as f:
                json.dump(counts_data, f)
            print(f"Saved counts after processing {dataset} {model}")
    
    print("Computation complete. All counts saved to behaviour_analysis_counts.json")



if __name__ == "__main__":
    main()