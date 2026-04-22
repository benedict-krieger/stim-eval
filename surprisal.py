import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers,torch
from wordfreq import zipf_frequency
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path




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


def bpe_split(word, tokenizer, bos_pad):

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
        **{col: row[col] for col in row.index},  
        'word_position': list(range(len(words))),  
        'word': words,  
        f'{surp_id}': surprisals,  
        'is_target': [True if i == last_occurrence else False for i in range(len(words))],
        'word_freq': word_freqs,
        'word_length': word_lengths  
    })

def get_word_freqs(word_list, lang):
    zipf_freqs = [zipf_frequency(w, lang, 'large') for w in word_list]
    return zipf_freqs


def merge_surprisal(user,exp):

    results_dir = Path("users")/ user / exp / "results" / "llm-surprisal"
    print(f"Merging surprisal output in {results_dir}")
    files = [f for f in results_dir.glob("*.tsv") if "_merged.tsv" not in f.name] # ignore merged files
    
    if len(files) < 2:
        print("Nothing to merge yet.")
        return
    
    
    final_df = pd.read_csv(files[0], sep='\t')

    print(f"Shape before merging: {final_df.shape}")

    for f in files[1:]:
        next_df = pd.read_csv(f, sep='\t')
        
        model_col = [c for c in next_df.columns if c.endswith("_surp")][0]
        
        join_keys = ['ItemNum', 'Condition' ,'word_position', 'word']
        
        final_df = final_df.merge(
            next_df[join_keys + [model_col]], 
            on=join_keys,
            how='left'
        )

    print(f"Shape after merging: {final_df.shape}")

    out_file = results_dir / f"{exp}_merged.tsv"
    final_df.to_csv(out_file, sep='\t', index=False)



#######################
#### Density plots ####
#######################

def add_vlines(df, col_name, surp_id, c_palette):
    '''
    Adding vertical mean lines to density plots. 
    '''
    col_vals = list(df[col_name].unique())
    for i, val in enumerate(col_vals):
        df_val = df[df[col_name] == val]
        mean = df_val[surp_id].mean()
        
        plt.axvline(mean, c=c_palette[i], linestyle='--', alpha=0.7)



def kde_plot_conditions(df, surp_id, outpath,
                        c_palette='husl', xlim=None, ylim=None,
                        title=None, show_legend=True):

    unique_conds = df['Condition'].unique()
    num_conds = len(unique_conds)

    c_palette = sns.color_palette(c_palette, n_colors=num_conds)

    plt.figure(figsize=(6, 4)) 
    sns.set(style='darkgrid')

    plot = sns.kdeplot(data=df,
                       x=surp_id,
                       hue='Condition',
                       palette=c_palette,
                       clip=(0, xlim) if xlim else None,
                       fill=True,
                       legend=show_legend
                       )
    
                            
    plot.set_xlabel("Surprisal")
    if xlim: plot.set_xlim(0, xlim)
    plot.set_ylabel("Density", fontsize=11)
    if ylim: plot.set_ylim(0, ylim) 
    plot.set_title(title)

    add_vlines(df, 'Condition', surp_id, c_palette)
    
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()

###########################################################################################
###########################################################################################