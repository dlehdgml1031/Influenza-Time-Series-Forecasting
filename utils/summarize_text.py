import os
import sys
import re
from typing import List, Dict, Any, Tuple

import pandas as pd

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

def _load_text_data(
    input_text_flag:str,
):  
    assert input_text_flag in ['News', 'Patent', 'Abstract', 'Announcement']
    
    if input_text_flag == 'News':
        text_df = pd.read_csv('./data/Text/News.csv', index_col = 0).set_index('Weekly_Date_Custom')

    elif input_text_flag == 'Patent':
        text_df = pd.read_csv('./data/Text/Patent.csv', index_col = 0).set_index('Weekly_Date_Custom')
        text_df.drop_duplicates(['PATENT_ABSTC'], keep = 'first', inplace=True)
        text_df.sort_index(inplace=True)
        
        
    
    

def _generate_summarization_prompt(
    input_text: str,
    input_text_flag:str,
):
    
    assert input_text_flag in ['News', 'Patent', 'Abstract', 'Announcement']
    
    if input_text_flag == 'News':
        summarization_prompt = """
        
        
        """
    
    elif input_text_flag == 'Patent':
        summarization_prompt = """
        
        Please summarize the following noisy but possible news data extracted from
        web page HTML, and extract keywords of the news. The news text can be very noisy due to it is HTML extraction. Give formatted
        answer such as Summary: ..., Keywords: ... The news is supposed to be for {symbol} stock. You may put ’N/A’ if the noisy text does
        not have relevant information to extract.
        
        Patent: {input_text}
        
        """
    
    
    return summarization_prompt

def generate_text_summarization(
    model_id: str,
    input_texts: List,
    input_text_flag:str,
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
    
    for input_text in input_texts:
        # Use dialogue template if chat_template_flag is True; otherwise, use input query directly
        if chat_template_flag:
            messages = [
                {
                    "role" : "user",
                    "content" : input_text
                }
            ]
            
            prompt = tokenizer.apply_chat_template(
                conversation = messages,
                tokenize = False,
                add_generation_prompt = True,
            )
            
            input_ids = tokenizer(prompt, return_tensors = "pt").to(device)
        else:
            input_ids = tokenizer(input_text, return_tensors="pt").to(device)
        
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