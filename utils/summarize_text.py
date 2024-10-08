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
    
    text_df = pd.read_csv(f'./data/Text_Input_New/{input_text_flag.lower()}_for_input.csv', index_col = 0).set_index('Weekly_Date_Custom')
    text_df.columns = [input_text_flag]
    text_df.fillna('N/A', inplace = True)
    
    return text_df
   

def _generate_summarization_prompt(
    input_text: str,
    input_text_flag:str,
):
    
    assert input_text_flag in ['News', 'Patent', 'Abstract']
    
    if input_text_flag == 'News':
        if len(input_text) > 20_000:
            input_text = input_text[:20_000]
            
        summarization_prompt = f"""
        [Instructions]
        Please summarize the following noisy news title data. The data may contain multiple stitched news titles, and it's focused on infectious diseases. The summary should be concise, logically connected, start with 'Summary:', and written in a single paragraph of cohesive, continuous sentences. Avoid using bullet points or listing information separately. Instead, ensure that the summary reads as a fluid, connected narrative.

        Summarize the content using the format in the [Example Format]. If no relevant information is available for summarization, return 'N/A'.

        [Example Answer Format]
        Summary: sentence 1. sentence 2. sentence 3 ... sentence N.

        [Input News Title]
        News Title: {input_text}
        """


    elif input_text_flag == 'Patent':
        summarization_prompt = f"""
        Please summarize the following noisy patent text data. The patent text data can be very noisy because it is stitched together from multiple patents. The patent is supposed to be for Infectious diseases. 
        Summarize the answer in a single paragraph, ensuring that each sentence is logically connected. Only a summary is required, Give formatted answer such as Summary: ... .
        If there is no relevant information that can be summarized from the text, you can enter 'N/A'.
        
        [Example Format]
        Summary: sentence 1. sentence 2. sentence 3 ... .
        
        [Input]
        Patent: {input_text}
        
        """
        
    elif input_text_flag == 'Abstract':
        summarization_prompt = f"""
        Please summarize the following noisy paper abstract text data. The paper abstract text data can be very noisy because it is stitched together from multiple paper abstract. 
        Summarize the answer in a single paragraph, ensuring that each sentence is logically connected. Only a summary is required, Give formatted answer such as Summary: ... .
        If there is no relevant information that can be summarized from the text, you can enter 'N/A'.
        
        Paper Abstract: {input_text}
        
        
        """
    
    return summarization_prompt

def generate_text_summarization(
    model_id: str,
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
    
    # Load text data
    text_df = _load_text_data(input_text_flag)
    input_texts = text_df[input_text_flag].tolist()
    
    result_dict = {
        'Weekly_Date_Custom': text_df.index.to_list(),
        'Input_text': [],
        'Summary': [],
    }
    
    # For Llama_3_8B_Instruct model
    if 'Llama_3_8B_Instruct' in model_id:
        assert chat_template_flag == True # Check if chat template is enabled
        
        for input_text in input_texts:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert assistant specializing in summarizing information on infectious diseases. Your role is to provide concise and accurate summaries of news and complex texts, focusing on key details such as disease characteristics, transmission, symptoms, prevention, and treatment. Ensure your summaries are clear and useful for healthcare professionals and informed readers."
                },
                
                {
                    "role" : "user",
                    "content" : _generate_summarization_prompt(input_text, input_text_flag)
                }
            ]
            
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt = True,
                return_tensors = "pt",
            ).to(device)
            
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            # Generate answer
            outputs = llm_model.generate(
                input_ids,
                max_new_tokens = gen_token_len,
                eos_token_id = terminators,
                pad_token_id = tokenizer.eos_token_id,
                # do_sample = True,
                # temperature = 0.6,
                # top_p = 0.9,
            )
            
            response = outputs[0][input_ids.shape[-1]:]
            final_response = tokenizer.decode(response, skip_special_tokens=True)
            
            print(final_response, "\n\n")
            
            result_dict['Input_text'].append(input_text)
            result_dict['Summary'].append(final_response)
    
    # For other models
    else:
        for input_text in input_texts:
            # Use dialogue template if chat_template_flag is True; otherwise, use input query directly
            if chat_template_flag:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a highly knowledgeable assistant specialized in summarizing information related to infectious diseases. Your role is to provide concise, accurate, and clear summaries of complex texts, focusing on the most critical information about disease characteristics, transmission, symptoms, prevention, and treatment. Ensure that your summaries are accessible to healthcare professionals and informed individuals, maintaining a balance between detail and clarity."
                    },
                    
                    {
                        "role" : "user",
                        "content" : _generate_summarization_prompt(input_text, input_text_flag)
                    }
                ]
                
                prompt = tokenizer.apply_chat_template(
                    conversation = messages,
                    tokenize = False,
                    add_generation_prompt = True,
                )
                
                input_ids = tokenizer(prompt, return_tensors = "pt").to(device)
            else:
                input_ids = tokenizer(_generate_summarization_prompt(input_text, input_text_flag), return_tensors = "pt").to(device)
            
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
            
            print(outputs_decoded)
        
    pd.DataFrame(result_dict).to_csv(f"./data/Text_Summarization_New/{input_text_flag.lower()}_summarization.csv")
    
        
if __name__ ==  "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "8"
    
    model_dir = "/mnt/nvme01/huggingface/models/"
    model_id = 'MetaAI/Llama_3_8B_Instruct'
    input_text_flag = 'News'
    gen_token_len = 512
    chat_template_flag = True
    quantization_flag = False
    flash_attention_flag = False
    
    generate_text_summarization(
        model_id = model_dir + model_id,
        input_text_flag = input_text_flag,
        gen_token_len = gen_token_len,
        chat_template_flag = chat_template_flag,
        quantization_flag = quantization_flag,
        flash_attention_flag = flash_attention_flag
    )