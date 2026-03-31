import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers,torch
from wordfreq import zipf_frequency
import yaml
from pathlib import Path


def get_config(llm_name):
    '''Load model config'''
    with open(f"llms.yaml", "r") as f:
        llms = yaml.safe_load(f)
        llm_cfg = llms[llm_name]

    return llm_cfg


def chunkstring(string, length):

    '''
    Chunks string into sub-strings of specified max length.
    Returns list of chunks.
    '''

    return (list(string[0+i:length+i] for i in range(0, len(string), length)))


def get_surprisal(input_str, model, tokenizer, ws_ind, char_repl, bos_pad):

    '''
    Returns surprisal for the last word of a string.

    input_str (str): input string
    model (object): pre-loaded model instance
    tokenizer (object): pre-loaded tokenizer instance
    ws_ind (char): special character indicating initial whitespace when using convert_ids_to_tokens
    char_repl (bool) : whether character replacement is needed (unicode problems in some GPT-models)
    bos_pad (bool) : whether input sequence needs to be padded with the bos token

    '''

    model.eval()
    if hasattr(model.config, "max_position_embeddings"): # attribute that contains context size is dependent on model-specific config
        ctx_window = model.config.max_position_embeddings # Llama2 config
    elif hasattr(model.config, "n_positions"):
        ctx_window = model.config.n_positions # GPT config

    chunk_size = int(0.75*ctx_window) # chunk size based on LLM's context window size

    seq_chunks = chunkstring(input_str.split(),chunk_size) # returns chunks with words as items

    words, surprisals = [] , []

    for seq in seq_chunks:

        subword_tokens, subword_surprisals = [] , []
        
        inputs = tokenizer(seq, is_split_into_words=True,return_tensors="pt")

        if bos_pad:
            # Manually prepend BOS token if required (e.g., GerPT-2)
            bos_id_tensor = torch.tensor([[tokenizer.bos_token_id]])
            attn_mask_tensor = torch.tensor([[1]])
            
            input_ids = torch.cat((bos_id_tensor, inputs["input_ids"]), dim=1)
            attention_mask = torch.cat((attn_mask_tensor, inputs["attention_mask"]), dim=1)
        else:
            # Use inputs as-is (e.g., Llama/LeoLM usually handles BOS automatically or doesn't need manual padding)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

        # Move tensors to GPU
        model_inputs = {
            "input_ids": input_ids.to(model.device),
            "attention_mask": attention_mask.to(model.device)
        }
       
        
        with torch.no_grad():
            outputs = model(**model_inputs)
        
        logits = outputs.logits
        logits = logits.cpu()
        output_ids = model_inputs["input_ids"].squeeze(0)[1:]
        tokens = tokenizer.convert_ids_to_tokens(model_inputs["input_ids"].squeeze(0))[1:]
        index = torch.arange(0, output_ids.shape[0])

        index = torch.arange(0, output_ids.shape[0], device=logits.device)
        output_ids = output_ids.to(logits.device)  # Move to same device as logits

        surp = -1 * torch.log2(F.softmax(logits.float(), dim = -1).squeeze(0)[index, output_ids]) # Convert logits to FP32 just for this step, to prevent inf surprisal

        subword_tokens.extend(tokens)
        subword_surprisals.extend(surp.cpu().numpy().tolist())

        # Word surprisal
        i = 0
        temp_token = ""
        temp_surprisal = 0
        
        while i <= len(subword_tokens)-1:

            temp_token += subword_tokens[i]
            temp_surprisal += subword_surprisals[i]
            
            if i == len(subword_tokens)-1 or tokens[i+1].startswith(ws_ind):
                # remove start-of-token indicator
                words.append(temp_token[1:])
                surprisals.append(temp_surprisal)
                # reset temp token/surprisal
                temp_surprisal = 0
                temp_token = ""
            i += 1
    if char_repl:
        replace_dict = {'ÃĦ':'Ä','Ã¤':'ä','Ãĸ':'Ö','Ã¶':'ö','Ãľ':'Ü','Ã¼':'ü',
                        'ÃŁ':'ß','âĢľ':'“','âĢŀ':'„','Ãł':'à','ÃĢ':'À','Ã¡':'á',
                        'Ãģ':'Á','Ã¨':'è','ÃĪ':'È','Ã©':'é','Ãī':'É','Ã»':'û',
                        'ÃĽ':'Û','ÃŃ':'í','âĢĵ':'–','âĢĻ':'’'}
        for k in replace_dict.keys():
            words = [w.replace(k,replace_dict[k]) for w in words]

    return words, surprisals


def bpe_split(word, bos_pad):

    '''
    Test if a given (target) word is split by the tokenizer into multiple subwords.
    
    If the tested tokenizer automatically prepends the BOS token (bos_pad=False), length 2 indicates 
    BOS + single token, while length > 2 indicates multiple subwords.
    
    If the tokenizer does not prepend the BOS token (bos_pad=True), length > 1 indicates multiple subwords.
    '''

    encoded_w = tokenizer.encode(word)

    if len(encoded_w) > 2:
        bpe_split = 1
    elif bos_pad and (len(encoded_w) > 1):
        bpe_split = 1
    else:
        bpe_split = 0

    return bpe_split


def process_row(row, model, tokenizer, ws_ind, char_repl, bos_pad, surp_id, lang):
    
    words, surprisals = get_surprisal(row['Stimulus'], model, tokenizer, ws_ind, char_repl, bos_pad)
    word_freqs = get_word_freqs(words,lang)
    word_lengths = [len(w) for w in words]

    last_occurrence = max([j for j, w in enumerate(words) if w == row['Target']], default=-1)

    return pd.DataFrame({
        **{col: row[col] for col in df.columns},  
        'Word_Position': list(range(len(words))),  
        'Word': words,  
        f'{surp_id}': surprisals,  
        'Is_Target': [1 if i == last_occurrence else 0 for i in range(len(words))],
        'Word_Freq': word_freqs,
        'Word_Length': word_lengths  
    })

def get_word_freqs(word_list, lang):
    zipf_freqs = [zipf_frequency(w, lang, 'large') for w in word_list]
    return zipf_freqs

###########################################################################################
###########################################################################################

if __name__ == '__main__':

    all_models = ['leo13b','gerpt2','gerpt2-large','gpt2']
    #base_model_path = "/scratch/common_models"

    parser = argparse.ArgumentParser()
    parser.add_argument('-u','--user',help=f'User name, for which a sub-directory must exist')
    parser.add_argument('-d','--data',help = f'Input csv file, tab separated')
    parser.add_argument('-m','--model',help = f'Models:{all_models}')
    args = parser.parse_args()

    llm_name = args.model
    llm_cfg = get_config(llm_name)
    repo_id = llm_cfg['repo_id']
    surp_id = llm_cfg['surp_id']
    bpe_id = llm_cfg['bpe_id']
    ws_ind = llm_cfg['ws_ind']
    char_repl = llm_cfg['char_repl']
    bos_pad = llm_cfg['bos_pad']
    lang = llm_cfg['lang']



    print(f'Model id: {llm_name}')
    device_arg = "auto" if torch.cuda.is_available() else None
    dtype_arg = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(repo_id, add_prefix_space=True)
    model = AutoModelForCausalLM.from_pretrained(repo_id,device_map=device_arg,dtype=dtype_arg)
    print(f"Model on {model.device}")
    
    df = pd.read_csv(f'{args.user}/data/{args.data}', sep ="\t")
    filename = Path(args.data).stem
    long_df = df.apply(process_row, axis=1, args=(model,tokenizer,ws_ind,char_repl, bos_pad, surp_id, lang))
    long_df = pd.concat(long_df.values, ignore_index=True)
    results_path = f'{args.user}/results'
    Path(results_path).mkdir(parents=True, exist_ok=True)
    long_df.to_csv(f'{results_path}/{filename}_{llm_name}.csv', sep='\t', index = False)
