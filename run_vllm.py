import argparse
import aiohttp
from aiohttp import ClientTimeout, ClientError
import asyncio
import os
import re
import json
import random
from tqdm.asyncio import tqdm_asyncio
from eval.utils import (
    generate_completions,
    load_lm_and_tokenizer,
    load_dexperts_model_and_tokenizer,
    dynamic_import_function,
    ensure_dir
)
from eval.math_util import my_answer_extraction
from eval.math_equivalence import is_equiv
from eval.lcb_prompts import get_generic_question_template_answer 

async def call_vllm(session, prompt, model, url, retries=3, timeout=3600):
    """Call vLLM endpoint with retries and per-request timeout.

    Returns the generated text on success. On repeated failures raises the last
    exception so caller can decide what to do (we also catch exceptions in the
    caller to avoid un-retrieved task exceptions).
    """
    payload = {
        "model": model,
        "temperature": 0.6,
        "max_tokens": 2048,                 # <-- set max generation tokens
        "messages": [{"role": "user", "content": prompt}],
    }

    backoff = 1.0
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            # per-request timeout
            async with session.post(url, json=payload, timeout=ClientTimeout(total=timeout)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

        except asyncio.TimeoutError as e:
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            raise
        except ClientError as e:
            # aiohttp client errors (connection errors, etc.)
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            raise
        except Exception as e:
            # For any other unexpected exception, record and re-raise after retries
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            raise

    # If we get here, re-raise the last exception
    if last_exc:
        raise last_exc

async def run_vllm_concurrent(prompts, model):
    if model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B":
        url = "http://localhost:8002/v1/chat/completions"
    elif model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
        url = "http://localhost:8001/v1/chat/completions"
    elif model == "Qwen/Qwen2.5-14B-Instruct-1M":
        url = "http://localhost:8003/v1/chat/completions"
    else:
        raise ValueError(f"unsupported model = {model}")

    # Use a session and create real asyncio Tasks so we can robustly await and
    # catch per-task exceptions (prevents "Task exception was never retrieved").
    async with aiohttp.ClientSession() as session:
        async def run_single(idx, prompt):
            try:
                result = await call_vllm(session, prompt, model, url)
            except Exception as e:
                # Record the exception as a string so the caller can continue.
                result = f"__ERROR__:{type(e).__name__}:{e}"
            return idx, result

        tasks = [asyncio.create_task(run_single(idx, prompt)) for idx, prompt in enumerate(prompts)]

        results = [None] * len(prompts)

        # Await tasks as they complete, keeping progress bar updates.
        for task in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            idx, result = await task
            results[idx] = result

        return results


def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output


def main(args):
    random.seed(42)

    print("Loading data...")
    all_test_data = []
    with open(os.path.join(args.data_dir, "test.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            if "lcb" in args.data_dir.lower():
                example["question"] = example["question_content"]
                example["answer"] = ""
                all_test_data.append(example)
            else:
                example_question = example["question"]
                example_answer = example["answer"]
                all_test_data.append({
                    "question": example_question,
                    "answer": str(example_answer)
                })

    test_data = all_test_data
    if args.max_examples and len(all_test_data) > args.max_examples:
        test_data = all_test_data[:args.max_examples]

    output_file = os.path.join(args.save_dir, "predictions.jsonl")
    if os.path.exists(output_file):
        with open(output_file, 'r') as reader:
            curr_samples = len([l for l in reader])
        test_data = all_test_data[curr_samples:]
        print(f"skipping {curr_samples} examples")



    ensure_dir(args.save_dir)

    gpqa_instr = "Solve the following problem. You must choose only one option from A to D. Your final answer should be a single letter from A to D, in the form \\boxed{answer}, at the end of your response.\n\n"
    math_instr = "Solve the following math problem. Put your final answer within \\boxed{}.\n\n"
    code_instr = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"

    if "gpqa" in args.data_dir.lower():
        if "icl" in args.data_dir.lower():
            prompt_prefix = open("long_cot_prompt_gpqa.txt", 'r').read() + gpqa_instr
        else:
            prompt_prefix = gpqa_instr
        problem_type = "mutiple_choice"
    elif "icl" in args.data_dir.lower():
        prompt_prefix = open("long_cot_prompt.txt", 'r').read() + math_instr 
        problem_type = "math"
    elif "lcb" in args.data_dir.lower():
        prompt_prefix = code_instr
        problem_type = "code"
    else:
        prompt_prefix = math_instr
        problem_type = "math"
    # problem_type = "mutiple_choice" if "gpqa" in args.data_dir.lower() else "math"


    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    for example in test_data:
        if "lcb" in args.data_dir.lower():
            prompt = prompt_prefix + get_generic_question_template_answer(example)
        else:
            prompt = prompt_prefix + "Question: " + example["question"].strip()
        prompts.append(prompt)

    if len(prompts) > 0:
        with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
            fout.write(prompts[0])

    outputs = asyncio.run(run_vllm_concurrent(prompts, args.model_name_or_path))
    items = [{"question": prompt, "answer": example["answer"], "model_output": output, "prediction" : my_answer_extraction(output, problem_type=problem_type)} for prompt, output, example in zip(prompts, outputs, test_data)]

    with open(output_file, 'a') as writer:
        for item in items:
            writer.write(json.dumps(item)+'\n')

    with open(output_file, 'r') as reader:
        predictions = [my_answer_extraction(json.loads(line)["model_output"], problem_type=problem_type) for line in reader]
    # predictions = [my_answer_extraction(output) for output in outputs]
    targets = [example["answer"] for example in all_test_data]
    assert len(predictions) == len(targets), f"len(predictions) = {len(predictions)}, len(targets) = {len(targets)}"

    em_score = sum([int(is_equiv(pred, answer)) for pred, answer in zip(predictions, targets)]) / len(targets)

    print(f"Exact match : {em_score}")

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump({
            "exact_match": em_score
        }, fout, indent=4)





if __name__ == "__main__":
    def str_to_bool(value):
        if isinstance(value, bool):  # If already a bool, return it directly
            return value
        if value.lower() in ("true", "1", "yes", "t"):
            return True
        elif value.lower() in ("false", "0", "no", "f"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected (True/False).")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/gsm"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="initial alpha"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="DEPRECATED if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="max generation tokens"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
    )
    parser.add_argument(
        "--expert_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--anti_expert_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tokenizer_mapping_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--alpha_strategy",
        type=str,
        default=None
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--budget_forcing",
        type=str_to_bool,
        default=False,
        help="replace eos token with Wait"
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    args = parser.parse_args()

    main(args)
