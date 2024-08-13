import re

import os
import sys
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM

def _extract_question_answer(text: str):
    # Define regex patterns
    user_pattern = r'user\s+(.*)\s+assistant'
    assistant_pattern = r'assistant\s+(.*)'

    # Find user question
    user_match = re.search(user_pattern, text, re.DOTALL)
    question = user_match.group(1).strip() if user_match else None

    # Find assistant answer
    assistant_match = re.search(assistant_pattern, text, re.DOTALL)
    answer = assistant_match.group(1).strip() if assistant_match else None

    return question, answer

def generate_text_summary(
    model_id: str,
    input_querys: List,
    gen_token_len:int = 256,
    chat_template_flag: bool = True,
    quantization_flag: bool = False,
    flash_attention_flag: bool = False,
):  
    
    # Define the device if cuda is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Quantization config
    if quantization_flag:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_compute_dtype = torch.float16
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load LLM model
    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = model_id,
        torch_dtype = torch.float16,
        quantization_config = quantization_config if quantization_flag else None,
        # low_cpu_mem_usage = False, # use as much memory as we can
        # attn_implementation = attn_implementation, # use flash attention
    )
    
    if quantization_flag:
        pass
    else:
        llm_model.to(device)
    
    for input_query in input_querys:
            # Use dialogue template if chat_template_flag is True; otherwise, use input query directly
        if chat_template_flag:
            messages = [
                {
                    "role" : "user",
                    "content" : input_query
                }
            ]
            
            prompt = tokenizer.apply_chat_template(
                conversation = messages,
                tokenize = False,
                add_generation_prompt = True,
            )
            
            input_ids = tokenizer(prompt, return_tensors = "pt").to(device)
        else:
            input_ids = tokenizer(input_query, return_tensors="pt").to(device)
        
        # Remove token_type_ids
        input_ids.pop("token_type_ids", None)
        
        # Generate answer
        outputs = llm_model.generate(
            **input_ids,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id,
            max_new_tokens = gen_token_len, # maximum number of tokens to generate
        )
        
        outputs_decoded = tokenizer.decode(
            outputs[0],
            skip_special_tokens = True,
        )
        
        # Extract question and answer from the outputs_decoded
        q, a = _extract_question_answer(outputs_decoded)
        print(f'Q: {q}')
        print(f'A: {a}')
        
        return a