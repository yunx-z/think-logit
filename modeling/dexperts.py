"""
ThinkLogit: Decoding-time Experts for Controllable Text Generation

This module implements the ThinkLogit framework, which steers a base language model's
generations by combining its logits with those from an expert and anti-expert model:

    final_logits = base_logits + alpha * (expert_logits - antiexpert_logits)

Key Features:
- Supports both same-family (shared tokenizer) and cross-family (different tokenizers) setups
- Parallel model inference using ThreadPoolExecutor for efficiency
- Optional logging of top-k logits at each generation step

Reference: https://arxiv.org/abs/2105.03023
"""

import sys
import time
import os
import json
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
import torch.nn.functional as F
from collections import defaultdict
from modeling.utils import top_k_top_p_filtering
from concurrent.futures import ThreadPoolExecutor, as_completed

# Llama-2 chat format markers
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
    """
    ThinkLogit model combining a base LM with expert and anti-expert models.
    
    The generation follows:
        final_logits = base_logits + alpha * (expert_logits - antiexpert_logits)
    
    This allows steering the base model toward expert-like behavior while
    avoiding anti-expert behavior, without fine-tuning the base model.
    """
    
    def __init__(
        self,
        base_model_name_or_path: str,
        expert_model_name_or_path: str,
        antiexpert_model_name_or_path: str,
        base_tokenizer: PreTrainedTokenizer,
        expert_tokenizer: Optional[PreTrainedTokenizer] = None,
        tokenizer_mapping_path: Optional[str] = None,
        system_prompt: str = None,
        alpha: float = 1.0,
        chat_response_prefix: str = None,
        model_kwargs: Dict[str, Any] = None,
        log_file: Optional[str] = None,
    ):
        """
        Initialize ThinkLogit with base, expert, and anti-expert models.
        
        Args:
            base_model_name_or_path: HuggingFace model ID or path for the base model
            expert_model_name_or_path: HuggingFace model ID or path for the expert model
            antiexpert_model_name_or_path: HuggingFace model ID or path for the anti-expert
            base_tokenizer: Tokenizer for the base model
            expert_tokenizer: Tokenizer for expert/antiexpert (required for cross-family setup)
            tokenizer_mapping_path: JSON file mapping base tokens to expert tokens (cross-family)
            system_prompt: Optional system prompt for chat-formatted models
            alpha: Scaling factor for the expert-antiexpert delta (default: 1.0)
            chat_response_prefix: Prefix to prepend to model responses (e.g., "Answer:")
            model_kwargs: Additional kwargs passed to AutoModelForCausalLM.from_pretrained
            log_file: Path to write per-step logit debugging information
        """

        # Store model paths
        self.base_model_name_or_path = base_model_name_or_path
        self.expert_model_name_or_path = expert_model_name_or_path
        self.antiexpert_model_name_or_path = antiexpert_model_name_or_path
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

        # Load base model immediately; expert/antiexpert are lazy-loaded on first generate() call
        self.base = self.load_model(self.base_model_name_or_path, self.model_kwargs)
        self.expert = None
        self.antiexpert = None

        # Tokenizer setup: for cross-family models, expert uses its own tokenizer
        self.base_tokenizer = base_tokenizer
        self.expert_tokenizer = expert_tokenizer if expert_tokenizer is not None else base_tokenizer
        self.tokenizer = self.base_tokenizer  # Alias for backward compatibility

        self.device = self.base.device
        self.chat_response_prefix = chat_response_prefix

        # Chat format configuration (currently disabled)
        self.use_chat_format_for_expert = False
        if self.use_chat_format_for_expert:
            self.chat_prefix = "[INST]"
            self.chat_suffix = "[/INST]"
            if system_prompt:
                self.chat_prefix += f"{B_SYS}{system_prompt}{E_SYS}"
            if self.chat_response_prefix:
                self.chat_suffix += f" {chat_response_prefix}"

        # Scaling factor for expert-antiexpert logit difference
        self.alpha = 1.0

        self.log_file = log_file
        if self.log_file and os.path.exists(self.log_file):
             try:
                 os.remove(self.log_file)
             except OSError as e:
                 print(f"Warning: Could not remove existing log file {self.log_file}: {e}", file=sys.stderr)

        # Cross-family token mapping: maps base tokenizer IDs to expert tokenizer IDs
        # This enables ThinkLogit to work with models that use different tokenizers
        # (e.g., Qwen base with Gemma expert). The mapping is stored as a tensor
        # where token_id_map[base_id] = expert_id, or -1 if no mapping exists.
        self.token_id_map = None
        if tokenizer_mapping_path:
            print("Cross-family mode enabled. Pre-processing token map...")
            if expert_tokenizer is None:
                raise ValueError("An expert_tokenizer must be provided for cross-family setup.")

            with open(tokenizer_mapping_path, 'r', encoding='utf-8') as f:
                token_map_str = json.load(f)

            # Initialize mapping tensor with -1 (unmapped) for all base vocab tokens
            self.token_id_map = torch.full((len(self.base_tokenizer),), -1, dtype=torch.long)

            # Convert token strings to IDs for both tokenizers
            base_tokens = list(token_map_str.keys())
            expert_tokens = list(token_map_str.values())
            base_token_ids = self.base_tokenizer.convert_tokens_to_ids(base_tokens)
            expert_token_ids = self.expert_tokenizer.convert_tokens_to_ids(expert_tokens)

            # Only keep mappings where both tokens exist in their respective vocabularies
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
        """Load a HuggingFace model and set it to eval mode."""
        if model_name_or_path:
            print(f"Loading model: {model_name_or_path}")
            a_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **_model_kwargs
            )
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


    def _run_model_forward(self, model, input_ids, model_kwargs):
        """Run a single forward pass through a model (used for parallel execution)."""
        if model is None:
            return None
        inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
        return outputs

    def _apply_template_and_tokenize(self, tokenizer, prompts, add_special_tokens=True):
        # Step 1: Convert each prompt into a chat-template string
        templated_texts = []
        for p in prompts:
            messages = [
                {"role": "system", "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
                {"role": "user", "content": p}
            ]

            # get raw text (not tokenized) — this ensures we can batch-pad
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,                # VERY IMPORTANT
                add_generation_prompt=True
            )
            templated_texts.append(text)

        # Step 2: Tokenize all at once → consistent padding
        tokenized = tokenizer(
            templated_texts,
            padding="longest",                # same behavior as your original call
            return_tensors="pt",
            add_special_tokens=add_special_tokens
        )

        # Move everything to device
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        return tokenized

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
        """
        Generate text using ThinkLogit decoding.
        
        At each step, computes:
            final_logits = base_logits + alpha * (expert_logits - antiexpert_logits)
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            do_sample: If True, sample from distribution; else greedy decode
            top_p: Nucleus sampling threshold (only used if do_sample=True)
            temperature: Sampling temperature (only used if do_sample=True)
            logits_processor: Optional HuggingFace logits processor
            stopping_criteria: Optional HuggingFace stopping criteria
            return_logits_for_analysis: If True, return analysis data with logits
            budget_forcing: If True, replace EOS tokens with "Wait" to force longer generation
            
        Returns:
            Generated token IDs [batch_size, seq_len + num_generated]
        """
        if input_ids is None:
            raise ValueError("input_ids must be provided.")

        # Prepare model kwargs for each model
        input_ids = input_ids.to(self.device)
        base_kwargs = kwargs.copy()
        expert_kwargs = kwargs.copy()
        antiexpert_kwargs = kwargs.copy()
        base_kwargs["model_name"] = self.base_model_name_or_path
        expert_kwargs["model_name"] = self.expert_model_name_or_path
        antiexpert_kwargs["model_name"] = self.antiexpert_model_name_or_path
        
        # Decode prompts for potential re-tokenization (cross-family or special templates)
        prompts = self.base_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Apply chat template for EXAONE models
        if "EXAONE" in self.base_model_name_or_path:
            template_tokenized = self._apply_template_and_tokenize(self.base_tokenizer, prompts)
            input_ids = template_tokenized["input_ids"]
            base_kwargs["attention_mask"] = template_tokenized["attention_mask"]

        # Lazy-load expert and antiexpert models on first use
        needs_experts = self.expert_model_name_or_path or self.antiexpert_model_name_or_path
        if needs_experts:
            if self.expert is None and self.expert_model_name_or_path:
                self.expert = self.load_model(self.expert_model_name_or_path, self.model_kwargs)
            if self.antiexpert is None and self.antiexpert_model_name_or_path:
                self.antiexpert = self.load_model(self.antiexpert_model_name_or_path, self.model_kwargs)

        # Prepare expert model inputs
        # For cross-family setups, we need to tokenize the prompt with the expert's tokenizer
        # For same-family setups, expert and base share the same input_ids
        if self.expert:
            if self.token_id_map is not None:
                # Cross-family: re-tokenize prompt with expert tokenizer
                expert_inputs = self.expert_tokenizer(prompts, return_tensors="pt", padding="longest").to(self.device)
                expert_input_ids = expert_inputs.input_ids
                expert_kwargs['attention_mask'] = expert_inputs.attention_mask
                antiexpert_kwargs['attention_mask'] = expert_inputs.attention_mask.clone()
            else:
                # Same-family: expert uses same input_ids as base
                expert_input_ids = input_ids
                if 'attention_mask' in base_kwargs:
                    if 'attention_mask' not in expert_kwargs:
                        expert_kwargs['attention_mask'] = base_kwargs['attention_mask'].clone()
                    antiexpert_kwargs['attention_mask'] = expert_kwargs['attention_mask'].clone()

        # Track which sequences are still generating
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        
        # Model-specific EOS token IDs for stopping generation
        if "Qwen" in self.base_model_name_or_path or "simplescaling" in self.base_model_name_or_path:
            stop_id_sequences = [151643, 151645]  # Qwen EOS tokens
        elif "Llama" in self.base_model_name_or_path:
            stop_id_sequences = [128001, 128009]  # Llama EOS tokens
        elif "olmo" in self.base_model_name_or_path.lower():
            stop_id_sequences = [100257, 100265]  # OLMo EOS tokens
        elif "EXAONE" in self.base_model_name_or_path:
            stop_id_sequences = [361, 42, 2]      # EXAONE EOS tokens
        elif "gemma" in self.base_model_name_or_path:
            stop_id_sequences = [1, 106]          # Gemma EOS tokens
        else:
            raise ValueError(f"{self.base_model_name_or_path} is missing stop_id_sequences")

        eos_token_id_tensor = torch.tensor(stop_id_sequences, device=input_ids.device)
        last_tokens = input_ids[:, -1].clone()
        repeat_counts = torch.zeros_like(unfinished_sequences)

        if return_logits_for_analysis:
            analysis_data = defaultdict(list)

        def write_log(step: int, base_logits_1d: torch.Tensor,
                      dexperts_logits_1d: Optional[torch.Tensor], next_token: str):
            if base_logits_1d is None or not self.log_file:
                return
            base_top10_tokens, base_top10_vals = _get_topk_info(base_logits_1d, self.base_tokenizer, k=10)
            if dexperts_logits_1d is not None:
                # ThinkLogit logits are in the base vocab space, so use base_tokenizer
                dexperts_top10_tokens, dexperts_top10_vals = _get_topk_info(dexperts_logits_1d, self.base_tokenizer, k=10)
            else:
                dexperts_top10_tokens, dexperts_top10_vals = None, None
            log_obj = {
                "step": step, "alpha": self.alpha, "next_token": next_token,
                "base_top10_tokens": base_top10_tokens, "dexperts_top10_tokens": dexperts_top10_tokens,
                "base_top10_logits": base_top10_vals, "dexperts_top10_logits": dexperts_top10_vals,
            }
            try:
                 with open(self.log_file, "a", encoding="utf-8") as f:
                     f.write(json.dumps(log_obj) + "\n")
            except Exception as e:
                 print(f"Error writing to log file {self.log_file}: {e}", file=sys.stderr)

        gen_steps = 0

        # Run models in parallel using ThreadPoolExecutor for efficiency
        num_workers = 1 + (1 if self.expert else 0) + (1 if self.antiexpert else 0)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            while gen_steps < max_new_tokens:
                # ===== Step 1: Parallel forward pass through all models =====
                futures = {}
                futures['base'] = executor.submit(self._run_model_forward, self.base, input_ids, base_kwargs)
                if self.expert:
                    futures['expert'] = executor.submit(self._run_model_forward, self.expert, expert_input_ids, expert_kwargs)
                if self.antiexpert:
                    futures['antiexpert'] = executor.submit(self._run_model_forward, self.antiexpert, expert_input_ids, antiexpert_kwargs)

                # Collect results from parallel execution
                results = {name: future.result() for name, future in futures.items()}
                base_outputs, expert_outputs, antiexpert_outputs = results.get('base'), results.get('expert'), results.get('antiexpert')
                if base_outputs is None:
                    raise RuntimeError("Base model forward pass failed.")

                # Extract logits for the next token position
                base_next_token_logits = base_outputs.logits[..., -1, :]
                expert_next_token_logits = expert_outputs.logits[..., -1, :] if expert_outputs else None
                antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :] if antiexpert_outputs else None

                # ===== Step 2: Compute ThinkLogit logits =====
                # Formula: final = base + alpha * (expert - antiexpert)
                dexperts_next_token_logits = None
                effective_alpha = self.alpha

                if effective_alpha != 0 and expert_next_token_logits is not None and antiexpert_next_token_logits is not None:
                    dexperts_next_token_logits = base_next_token_logits.clone()
                    
                    if self.token_id_map is not None:
                        # Cross-family: map expert logit deltas to base vocabulary space
                        delta_logits_expert_space = expert_next_token_logits - antiexpert_next_token_logits
                        delta_logits_base_space = torch.zeros_like(base_next_token_logits)

                        # Get indices of tokens that have valid mappings
                        mapped_base_indices = (self.token_id_map != -1).nonzero(as_tuple=True)[0]
                        mapped_expert_indices = self.token_id_map[mapped_base_indices]

                        # Gather deltas from expert space and scatter into base space
                        delta_for_mapped_tokens = torch.gather(
                            delta_logits_expert_space, -1,
                            mapped_expert_indices.expand(delta_logits_expert_space.shape[0], -1)
                        )
                        delta_for_mapped_tokens = delta_for_mapped_tokens.to(delta_logits_base_space.dtype)
                        delta_logits_base_space.scatter_(
                            -1,
                            mapped_base_indices.expand(delta_logits_base_space.shape[0], -1),
                            delta_for_mapped_tokens
                        )

                        dexperts_next_token_logits += effective_alpha * delta_logits_base_space
                    else:
                        # Same-family: directly add logit deltas
                        common_vocab = min(
                            base_next_token_logits.shape[-1],
                            expert_next_token_logits.shape[-1],
                            antiexpert_next_token_logits.shape[-1]
                        )
                        delta_logits = expert_next_token_logits[..., :common_vocab] - antiexpert_next_token_logits[..., :common_vocab]
                        dexperts_next_token_logits[..., :common_vocab] += effective_alpha * delta_logits

                    next_token_logits = dexperts_next_token_logits
                else:
                    # No experts available, use base logits only
                    next_token_logits = base_next_token_logits

                # ===== Step 3: Apply sampling/decoding =====
                if logits_processor:
                    next_token_logits = logits_processor(input_ids, next_token_logits)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                if top_p < 1.0:
                    next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)
                    
                probs = F.softmax(next_token_logits, dim=-1)
                if do_sample:
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # Budget forcing: replace EOS with "Wait" to encourage longer reasoning
                if budget_forcing:
                    eos_id = 151643  # Qwen EOS
                    wait_id = self.base_tokenizer.convert_tokens_to_ids("Wait")
                    if wait_id is None or wait_id == self.base_tokenizer.unk_token_id:
                        wait_id = self.base_tokenizer.convert_tokens_to_ids(" wait")
                    if wait_id is None:
                        raise ValueError("Could not find a valid token ID for 'Wait' in base tokenizer")
                    next_tokens = torch.where(next_tokens == eos_id, torch.full_like(next_tokens, wait_id), next_tokens)

                # Replace next_tokens with pad for finished sequences
                next_tokens = next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)
                
                # Detect and handle repetitive generation (force stop after 10 repeats)
                same_as_last = (next_tokens == last_tokens) & (unfinished_sequences == 1)
                repeat_counts = torch.where(same_as_last, repeat_counts + 1, torch.ones_like(repeat_counts))
                hit_limit = (repeat_counts >= 10) & (unfinished_sequences == 1)
                if hit_limit.any():
                    next_tokens = torch.where(hit_limit, torch.full_like(next_tokens, self.tokenizer.eos_token_id), next_tokens)
                    unfinished_sequences = unfinished_sequences * (~hit_limit)
                last_tokens = next_tokens.clone()

                next_token_str = self.base_tokenizer.decode(next_tokens[0])
                write_log(step=gen_steps, base_logits_1d=base_next_token_logits[0].detach().cpu(), dexperts_logits_1d=dexperts_next_token_logits[0].detach().cpu() if dexperts_next_token_logits is not None else None, next_token=next_token_str)
                
                if return_logits_for_analysis:
                    pass  # Analysis data collection (if enabled)

                # ===== Step 4: Update sequences for next iteration =====
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                
                if self.expert:
                    if self.token_id_map is not None:
                        # Cross-family: map generated token to expert vocabulary
                        next_tokens_for_expert = self.token_id_map[next_tokens]
                        unk_token_id = self.expert_tokenizer.unk_token_id or self.expert_tokenizer.pad_token_id
                        next_tokens_for_expert[next_tokens_for_expert == -1] = unk_token_id
                        expert_input_ids = torch.cat([expert_input_ids, next_tokens_for_expert[:, None]], dim=-1)
                    else:
                        # Same-family: use same token
                        expert_input_ids = torch.cat([expert_input_ids, next_tokens[:, None]], dim=-1)


                # Update KV caches and attention masks for next iteration
                if base_outputs:
                    base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
                if expert_outputs:
                    expert_kwargs = self._update_model_kwargs_for_generation(expert_outputs, expert_kwargs)
                if antiexpert_outputs:
                    antiexpert_kwargs = self._update_model_kwargs_for_generation(antiexpert_outputs, antiexpert_kwargs)

                # Check for EOS tokens and update unfinished sequences
                if eos_token_id_tensor is not None:
                    is_eos = next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                    unfinished_sequences = unfinished_sequences.mul(is_eos)
                    if unfinished_sequences.max() == 0:
                        break
                        
                if stopping_criteria is not None and stopping_criteria(input_ids, None):
                    break

                gen_steps += 1
       
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
