import sys
import time
import random
import os
import json
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
import torch.nn.functional as F
from collections import defaultdict
from modeling.utils import top_k_top_p_filtering
# Import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed

B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def _get_topk_info(logits: torch.Tensor, tokenizer: PreTrainedTokenizer, k: int = 10):
    """
    Returns the top-k tokens and logits (as Python lists) for inspection.
    Assumes logits is 1D, i.e. shape [vocab_size].
    """
    top_values, top_indices = torch.topk(logits, k)
    top_values = top_values.tolist()
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
    return top_tokens, top_values

class DExpertsLlama:
    def __init__(
        self,
        base_model_name_or_path: str,
        expert_model_name_or_path: str,
        antiexpert_model_name_or_path: str,
        base_tokenizer: PreTrainedTokenizer,
        expert_tokenizer: Optional[PreTrainedTokenizer] = None,
        tokenizer_mapping_path: Optional[str] = None,
        system_prompt: str = None,
        alpha: float = 1.0, # Note: alpha is mostly controlled by alpha_strategy now
        chat_response_prefix: str = None,
        model_kwargs: Dict[str, Any] = None,
        log_file: Optional[str] = None,
        alpha_strategy: str = None,
    ):
        """
        chat_response_prefix: For llama chat models, it can be helpful for the response
        to start with a certain prefix to constrain the generation to directly answer
        the question. This makes evaluation on MC datasets easier.
        """

        self.base_model_name_or_path = base_model_name_or_path
        self.expert_model_name_or_path = expert_model_name_or_path
        self.antiexpert_model_name_or_path = antiexpert_model_name_or_path
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

        self.base = self.load_model(self.base_model_name_or_path, self.model_kwargs)
        self.expert = None # Lazy load in generate if needed
        self.antiexpert = None # Lazy load in generate if needed

        self.base_tokenizer = base_tokenizer
        # --- CROSS-FAMILY: Use expert_tokenizer if provided, else default to base_tokenizer for same-family setup
        self.expert_tokenizer = expert_tokenizer if expert_tokenizer is not None else base_tokenizer
        self.tokenizer = self.base_tokenizer # For backward compatibility with any methods using self.tokenizer

        self.device = self.base.device
        self.chat_response_prefix = chat_response_prefix

        # Llama chat experts need different formatting
        self.use_chat_format_for_expert = bool(expert_model_name_or_path and 'chat' in expert_model_name_or_path.lower())

        if self.use_chat_format_for_expert:
            self.chat_prefix = "[INST]"
            self.chat_suffix = "[/INST]"
            if system_prompt:
                self.chat_prefix += f"{B_SYS}{system_prompt}{E_SYS}"
            if self.chat_response_prefix:
                self.chat_suffix += f" {chat_response_prefix}"

        # State machine variables
        self.phase = "S1_ZERO"
        self.phase_step_count = 0
        self.alpha = 1.0

        self.log_file = log_file
        if self.log_file and os.path.exists(self.log_file):
             try:
                 os.remove(self.log_file)
             except OSError as e:
                 print(f"Warning: Could not remove existing log file {self.log_file}: {e}", file=sys.stderr)

        self.alpha_strategy = alpha_strategy
        self.MAX_EPISODE = 3

        # --- CROSS-FAMILY: Pre-process token mapping for efficient lookup
        self.token_id_map = None
        if tokenizer_mapping_path:
            print("Cross-family mode enabled. Pre-processing token map...")
            if expert_tokenizer is None:
                raise ValueError("An expert_tokenizer must be provided for cross-family setup.")

            with open(tokenizer_mapping_path, 'r', encoding='utf-8') as f:
                token_map_str = json.load(f)

            # NEW, CORRECTED LINE
            self.token_id_map = torch.full((len(self.base_tokenizer),), -1, dtype=torch.long)

            base_tokens = list(token_map_str.keys())
            expert_tokens = list(token_map_str.values())

            base_token_ids = self.base_tokenizer.convert_tokens_to_ids(base_tokens)
            expert_token_ids = self.expert_tokenizer.convert_tokens_to_ids(expert_tokens)

            # Filter out tokens that might not be found (e.g., None for special tokens)
            valid_indices = [
                i for i, (b_id, e_id) in enumerate(zip(base_token_ids, expert_token_ids))
                if b_id is not None and e_id is not None
            ]

            if valid_indices:
                valid_base_ids = torch.tensor([base_token_ids[i] for i in valid_indices], dtype=torch.long)
                valid_expert_ids = torch.tensor([expert_token_ids[i] for i in valid_indices], dtype=torch.long)
                self.token_id_map[valid_base_ids] = valid_expert_ids

            self.token_id_map = self.token_id_map.to(self.device)
            print("Token map pre-processing complete.")


    def load_model(self, model_name_or_path, _model_kwargs):
        if model_name_or_path:
            print(f"Loading model: {model_name_or_path}")
            a_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **_model_kwargs
            )
            # print(model_name_or_path, "max_position_embeddings:", a_model.config.max_position_embeddings, flush=True)
            return a_model.eval()
        else:
            return None

    def _get_tokenized_chat_inputs(self, base_input_ids: torch.Tensor, target_tokenizer: PreTrainedTokenizer):
        """
        Decodes base input_ids and re-encodes with chat formatting using the target_tokenizer.
        """
        # --- CROSS-FAMILY: Always decode using the base tokenizer to get the original prompt string.
        prompts = self.base_tokenizer.batch_decode(base_input_ids, skip_special_tokens=True)

        cleaned_prompts = []
        if self.chat_response_prefix:
            for p in prompts:
                if self.chat_response_prefix in p:
                    p = p.replace(self.chat_response_prefix, '').rstrip()
                cleaned_prompts.append(p)
        else:
            cleaned_prompts = prompts

        chat_prompts = [f'{self.chat_prefix} {p} {self.chat_suffix}' for p in cleaned_prompts]
        # --- CROSS-FAMILY: Encode using the specified target_tokenizer.
        chat_inputs = target_tokenizer(
            chat_prompts, padding="longest", return_tensors="pt", add_special_tokens=True
        )
        chat_inputs = {k: v.to(self.device) for k, v in chat_inputs.items()}
        return chat_inputs


    def _update_phase(self, overriding_event_occurred: bool, extra_prompt_appended: bool):
        if extra_prompt_appended:
            self.phase = "S1_FINAL"
            self.phase_step_count = 0
        elif self.phase == "S1_ZERO":
            if self.phase_step_count >= 100:
                self.phase = "S2_ONE"
                self.phase_step_count = 0
        elif self.phase == "S2_ONE":
            if overriding_event_occurred:
                self.phase = "S1_ZERO"
                self.phase_step_count = 0
        elif self.phase == "S2_ONE_COUNTDOWN":
             if self.phase_step_count >= 100:
                 self.phase = "S1_ZERO"
                 self.phase_step_count = 0
        if self.phase in ["S1_FINAL", "S1_ZERO"]:
            self.alpha = 0
        else:
            self.alpha = 1

    def _run_model_forward(self, model, input_ids, model_kwargs):
        if model is None:
            return None
        inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        with torch.no_grad():
             outputs = model(**inputs, return_dict=True)
        return outputs

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        logits_processor=None,
        stopping_criteria=None,
        return_logits_for_analysis: bool = False,
        budget_forcing: bool = False,
        **kwargs
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided.")

        input_ids = input_ids.to(self.device)
        base_kwargs = kwargs.copy()
        expert_kwargs = kwargs.copy()
        antiexpert_kwargs = kwargs.copy()
        base_kwargs["model_name"] = self.base_model_name_or_path
        expert_kwargs["model_name"] = self.expert_model_name_or_path
        antiexpert_kwargs["model_name"] = self.antiexpert_model_name_or_path

        needs_experts = self.expert_model_name_or_path or self.antiexpert_model_name_or_path
        if needs_experts:
            if self.expert is None and self.expert_model_name_or_path:
                self.expert = self.load_model(self.expert_model_name_or_path, self.model_kwargs)
            if self.antiexpert is None and self.antiexpert_model_name_or_path:
                self.antiexpert = self.load_model(self.antiexpert_model_name_or_path, self.model_kwargs)

        # --- Prepare Expert Inputs ---
        # --- CROSS-FAMILY: This block now handles the initial tokenization for both same-family and cross-family setups.
        if self.expert:
            if self.token_id_map is not None:
                # --- CROSS-FAMILY INITIALIZATION ---
                # Decode the base prompt and re-encode with the expert tokenizer.
                prompts = self.base_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                expert_inputs = self.expert_tokenizer(prompts, return_tensors="pt", padding="longest").to(self.device)
                expert_input_ids = expert_inputs.input_ids
                expert_kwargs['attention_mask'] = expert_inputs.attention_mask
                antiexpert_kwargs['attention_mask'] = expert_inputs.attention_mask.clone() # Antiexpert shares expert's tokenization

                if self.use_chat_format_for_expert:
                    # Re-run chat formatting on the base prompt, but tokenize with the expert tokenizer
                    chat_inputs = self._get_tokenized_chat_inputs(input_ids, self.expert_tokenizer)
                    expert_input_ids = chat_inputs['input_ids']
                    if 'attention_mask' in chat_inputs:
                        expert_kwargs['attention_mask'] = chat_inputs['attention_mask']
                        antiexpert_kwargs['attention_mask'] = chat_inputs['attention_mask'].clone()
            else:
                # --- SAME-FAMILY INITIALIZATION (Original Logic) ---
                expert_input_ids = input_ids
                if self.use_chat_format_for_expert:
                    chat_inputs = self._get_tokenized_chat_inputs(input_ids, self.base_tokenizer)
                    expert_input_ids = chat_inputs['input_ids']
                    if 'attention_mask' in chat_inputs:
                        expert_kwargs['attention_mask'] = chat_inputs['attention_mask']
                # Ensure attention masks are consistent
                if 'attention_mask' in base_kwargs:
                    if 'attention_mask' not in expert_kwargs:
                        expert_kwargs['attention_mask'] = base_kwargs['attention_mask'].clone()
                    antiexpert_kwargs['attention_mask'] = expert_kwargs['attention_mask'].clone()

        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        if "Instruct" in self.base_model_name_or_path or "simplescaling" in self.base_model_name_or_path or "Olmo-3" in self.base_model_name_or_path:
            # s1 is based on Qwen-Instruct
            if "Qwen" in self.base_model_name_or_path or "simplescaling" in self.base_model_name_or_path:
                stop_id_sequences = [151643, 151645]
            elif "Llama" in self.base_model_name_or_path:
                stop_id_sequences = [128001, 128009]
            elif "olmo" in self.base_model_name_or_path.lower():
                stop_id_sequences = [100257, 100265]
            elif "EXAONE" in self.base_model_name_or_path:
                stop_id_sequences = [361, 42, 2]
            else:
                raise ValueError(f"{args.base_model_name_or_path} is missing stop_id_sequences")
        else:
            stop_id_sequences = None

        eos_token_id_tensor = torch.tensor(stop_id_sequences, device=input_ids.device)
        last_tokens = input_ids[:, -1].clone()
        repeat_counts = torch.zeros_like(unfinished_sequences)

        if return_logits_for_analysis:
            analysis_data = defaultdict(list)

        def write_log(step: int, base_logits_1d: torch.Tensor,
                      dexperts_logits_1d: Optional[torch.Tensor], alpha: float, phase: str, episode: int, next_token: str):
            if base_logits_1d is None or not self.log_file:
                return
            base_top10_tokens, base_top10_vals = _get_topk_info(base_logits_1d, self.base_tokenizer, k=10)
            if dexperts_logits_1d is not None:
                # DExperts logits are in the base vocab space, so use base_tokenizer
                dexperts_top10_tokens, dexperts_top10_vals = _get_topk_info(dexperts_logits_1d, self.base_tokenizer, k=10)
            else:
                dexperts_top10_tokens, dexperts_top10_vals = None, None
            log_obj = {
                "step": step, "phase": phase, "episode": episode, "alpha": alpha, "next_token": next_token,
                "base_top10_tokens": base_top10_tokens, "dexperts_top10_tokens": dexperts_top10_tokens,
                "base_top10_logits": base_top10_vals, "dexperts_top10_logits": dexperts_top10_vals,
            }
            try:
                 with open(self.log_file, "a", encoding="utf-8") as f:
                     f.write(json.dumps(log_obj) + "\n")
            except Exception as e:
                 print(f"Error writing to log file {self.log_file}: {e}", file=sys.stderr)

        gen_steps = 0
        allowed_gen_steps = max_new_tokens
        extra_prompt_appended = False
        curr_episode = 0
        overriding_event_occurred = False
        self.phase = "S1_ZERO"
        self.phase_step_count = 0

        num_workers = 1 + (1 if self.expert else 0) + (1 if self.antiexpert else 0)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            while gen_steps < allowed_gen_steps:
                if extra_prompt_appended: self.alpha = 0
                else:
                    current_alpha_strategy = self.alpha_strategy if self.alpha_strategy else "constant"
                    if current_alpha_strategy.startswith("constant"): self.alpha = float(current_alpha_strategy.replace("constant", "") or 1.0)
                    elif current_alpha_strategy.startswith("warmup"): self.alpha = 0.0 if gen_steps < int(current_alpha_strategy.replace("warmup", "") or 100) else 1.0
                    elif current_alpha_strategy.startswith("cycle"): self.alpha = 1.0 if (gen_steps // int(current_alpha_strategy.replace("cycle", "") or 100)) % 2 != 0 else 0.0
                    elif current_alpha_strategy.startswith("2cycles"):
                        items = current_alpha_strategy.split('-'); T0 = int(items[1]) if len(items) > 1 else 400; T1 = int(items[2]) if len(items) > 2 else 100
                        self.alpha = 0.0 if gen_steps % (T0 + T1) < T0 else 1.0
                    elif current_alpha_strategy.startswith("random"): self.alpha = 1.0 if random.random() < float(current_alpha_strategy.replace("random", "") or 0.5) else 0.0
                    elif current_alpha_strategy == "override_annealing": self._update_phase(overriding_event_occurred, extra_prompt_appended)
                    elif current_alpha_strategy == "ppt": self.alpha = 0 if curr_episode > self.MAX_EPISODE else 1.0
                    else: self.alpha = 1.0

                futures = {}
                futures['base'] = executor.submit(self._run_model_forward, self.base, input_ids, base_kwargs)
                if self.expert:
                    futures['expert'] = executor.submit(self._run_model_forward, self.expert, expert_input_ids, expert_kwargs)
                if self.antiexpert:
                    # Antiexpert and Expert share the same tokenization, so use expert_input_ids
                    futures['antiexpert'] = executor.submit(self._run_model_forward, self.antiexpert, expert_input_ids, antiexpert_kwargs)

                results = {name: future.result() for name, future in futures.items()}
                base_outputs, expert_outputs, antiexpert_outputs = results.get('base'), results.get('expert'), results.get('antiexpert')
                if base_outputs is None: raise RuntimeError("Base model forward pass failed.")

                base_next_token_logits = base_outputs.logits[..., -1, :]
                expert_next_token_logits = expert_outputs.logits[..., -1, :] if expert_outputs else None
                antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :] if antiexpert_outputs else None

                dexperts_next_token_logits = None
                effective_alpha = self.alpha

                # --- CROSS-FAMILY: This block now handles both same-family and cross-family logit combination.
                if effective_alpha != 0 and expert_next_token_logits is not None and antiexpert_next_token_logits is not None:
                    dexperts_next_token_logits = base_next_token_logits.clone()
                    if self.token_id_map is not None:
                        # --- CROSS-FAMILY DEXPERTS ---
                        delta_logits_expert_space = expert_next_token_logits - antiexpert_next_token_logits
                        delta_logits_base_space = torch.zeros_like(base_next_token_logits)

                        mapped_base_indices = (self.token_id_map != -1).nonzero(as_tuple=True)[0]
                        mapped_expert_indices = self.token_id_map[mapped_base_indices]

                        delta_for_mapped_tokens = torch.gather(delta_logits_expert_space, -1, mapped_expert_indices.expand(delta_logits_expert_space.shape[0], -1))
                        delta_for_mapped_tokens = delta_for_mapped_tokens.to(delta_logits_base_space.dtype)
                        delta_logits_base_space.scatter_(-1, mapped_base_indices.expand(delta_logits_base_space.shape[0], -1), delta_for_mapped_tokens)

                        
                        dexperts_next_token_logits += effective_alpha * delta_logits_base_space
                    else:
                        # --- SAME-FAMILY DEXPERTS ---
                        common_vocab = min(base_next_token_logits.shape[-1], expert_next_token_logits.shape[-1], antiexpert_next_token_logits.shape[-1])
                        delta_logits = expert_next_token_logits[..., :common_vocab] - antiexpert_next_token_logits[..., :common_vocab]
                        dexperts_next_token_logits[..., :common_vocab] += effective_alpha * delta_logits

                    next_token_logits = dexperts_next_token_logits
                else: # alpha is 0 or experts are missing
                    next_token_logits = base_next_token_logits

                overriding_event_occurred = False
                if self.alpha_strategy == "override_annealing" and self.alpha != 0 and dexperts_next_token_logits is not None:
                     if torch.argmax(base_next_token_logits[0], dim=-1).item() != torch.argmax(dexperts_next_token_logits[0], dim=-1).item():
                         overriding_event_occurred = True

                if logits_processor: next_token_logits = logits_processor(input_ids, next_token_logits)
                if temperature != 1.0: next_token_logits = next_token_logits / temperature
                if top_p < 1.0: next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1) if do_sample else torch.argmax(next_token_logits, dim=-1)

                if budget_forcing:
                    eos_id, wait_id = 151643, self.base_tokenizer.convert_tokens_to_ids("Wait")
                    if wait_id is None or wait_id == self.base_tokenizer.unk_token_id: wait_id = self.base_tokenizer.convert_tokens_to_ids(" wait")
                    if wait_id is None: raise ValueError("Could not find a valid token ID for 'Wait' in base tokenizer")
                    next_tokens = torch.where(next_tokens == eos_id, torch.full_like(next_tokens, wait_id), next_tokens)

                next_tokens = next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)
                same_as_last = (next_tokens == last_tokens) & (unfinished_sequences == 1)
                repeat_counts = torch.where(same_as_last, repeat_counts + 1, torch.ones_like(repeat_counts))
                hit_limit = (repeat_counts >= 10) & (unfinished_sequences == 1)
                if hit_limit.any():
                    next_tokens = torch.where(hit_limit, torch.full_like(next_tokens, self.tokenizer.eos_token_id), next_tokens)
                    unfinished_sequences = unfinished_sequences * (~hit_limit)
                last_tokens = next_tokens.clone()

                next_token_str = self.base_tokenizer.decode(next_tokens[0])
                write_log(step=gen_steps, base_logits_1d=base_next_token_logits[0].detach().cpu(), dexperts_logits_1d=dexperts_next_token_logits[0].detach().cpu() if dexperts_next_token_logits is not None else None, alpha=self.alpha, phase=self.phase, episode=curr_episode, next_token=next_token_str)
                
                if return_logits_for_analysis: # (Omitted for brevity, logic remains the same)
                    pass

                # --- Update input_ids for next iteration ---
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                if self.expert:
                    # --- CROSS-FAMILY: This block handles updating expert_input_ids for both setups.
                    if self.token_id_map is not None:
                        # --- CROSS-FAMILY UPDATE ---
                        next_tokens_for_expert = self.token_id_map[next_tokens]
                        unk_token_id = self.expert_tokenizer.unk_token_id or self.expert_tokenizer.pad_token_id
                        next_tokens_for_expert[next_tokens_for_expert == -1] = unk_token_id
                        expert_input_ids = torch.cat([expert_input_ids, next_tokens_for_expert[:, None]], dim=-1)
                    else:
                        # --- SAME-FAMILY UPDATE ---
                        expert_input_ids = torch.cat([expert_input_ids, next_tokens[:, None]], dim=-1)


                if base_outputs: base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
                if expert_outputs: expert_kwargs = self._update_model_kwargs_for_generation(expert_outputs, expert_kwargs)
                if antiexpert_outputs: antiexpert_kwargs = self._update_model_kwargs_for_generation(antiexpert_outputs, antiexpert_kwargs)

                if eos_token_id_tensor is not None:
                    is_eos = next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                    unfinished_sequences = unfinished_sequences.mul(is_eos)
                    if unfinished_sequences.max() == 0: break
                if stopping_criteria is not None and stopping_criteria(input_ids, None): break
                if self.alpha_strategy == "ppt" and self.is_thinking_token(next_token_str): curr_episode += 1

                gen_steps += 1
                # if gen_steps == max_new_tokens and not extra_prompt_appended:
                #      extra_prompt = "\nI'm not allowed to think more so I have to conclude that the final answer is:"
                #      extra_input = self.base_tokenizer(extra_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(input_ids.device)
                #      extra_input = extra_input.expand(input_ids.shape[0], -1)
                #      input_ids = torch.cat([input_ids, extra_input], dim=-1)

                #      if self.expert:
                #          # --- CROSS-FAMILY: Append the appropriately tokenized prompt to expert inputs
                #          if self.token_id_map is not None:
                #             expert_extra_input = self.expert_tokenizer(extra_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(input_ids.device)
                #             expert_extra_input = expert_extra_input.expand(input_ids.shape[0], -1)
                #             expert_input_ids = torch.cat([expert_input_ids, expert_extra_input], dim=-1)
                #          else:
                #             expert_input_ids = torch.cat([expert_input_ids, extra_input], dim=-1)
                #      
                #      for current_kwargs in [base_kwargs, expert_kwargs, antiexpert_kwargs]:
                #          if "attention_mask" in current_kwargs and torch.is_tensor(current_kwargs["attention_mask"]):
                #              extra_len = extra_input.shape[1] if current_kwargs is base_kwargs else expert_extra_input.shape[1]
                #              extra_attention = torch.ones((current_kwargs["attention_mask"].shape[0], extra_len), device=input_ids.device, dtype=current_kwargs["attention_mask"].dtype)
                #              current_kwargs["attention_mask"] = torch.cat([current_kwargs["attention_mask"], extra_attention], dim=-1)

                #      allowed_gen_steps += 100
                #      extra_prompt_appended = True
                #      self.phase = "S1_FINAL"; self.phase_step_count = 0

                if self.alpha_strategy == "override_annealing": self.phase_step_count += 1
        
        if return_logits_for_analysis:
            # (Omitted for brevity, logic remains the same)
            return input_ids, analysis_data
        return input_ids

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        This version manually extends the attention_mask and calculates cache_position.
        It is known to work for Llama and Qwen models.
        """
        model_kwargs["past_key_values"] = outputs.past_key_values
        if "attention_mask" in model_kwargs and torch.is_tensor(model_kwargs["attention_mask"]):
            # Manually extend the attention mask
            attention_mask = model_kwargs["attention_mask"]
            new_mask = attention_mask.new_ones((attention_mask.shape[0], 1))
            model_kwargs["attention_mask"] = torch.cat([attention_mask, new_mask], dim=-1)

            # Explicitly set cache_position based on the new mask's length (zero-based index)
            model_kwargs["cache_position"] = torch.tensor([model_kwargs["attention_mask"].shape[1] - 1], device=outputs.logits.device)

        return model_kwargs


    def is_thinking_token(self, token):
        return token.strip() in ["Wait", "Alternatively", "Hmm"]
