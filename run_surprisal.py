import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers,torch
from wordfreq import zipf_frequency
import yaml
from pathlib import Path
import surprisal as surp

def get_llm_config(llm_name):
    '''Load model config'''
    with open("config/llms.yaml", "r") as f:
        llms = yaml.safe_load(f)
        llm_cfg = llms[llm_name]

    return llm_cfg

def get_study_config(filename):
    """
    Attempts to find plot settings based on the input filename.
    """
    with open("config/studies.yaml", "r") as f:
            all_configs = yaml.safe_load(f)
    
    # Look for the filename (e.g., 'test' from 'test.tsv')
    return all_configs.get(filename, {})

def get_plot_params(study_cfg, filename, llm_name):
    """
    Applies fallback logic using study_cfg dict.
    """
    return {
        'xlim': study_cfg.get('x_lim', None),
        'ylim': study_cfg.get('y_lim', None),
        'title': study_cfg.get('title', f"{filename} {llm_name} surprisal"),
        'c_palette': study_cfg.get('c_palette', "husl"),
        'c_labels': study_cfg.get('c_labels', None)
    }


if __name__ == '__main__':

    all_models = ['leo13b', 'llammlein7b', 'llammlein1b', 'llammlein120m', 'gerpt2', 'gerpt2-large']
    #base_model_path = "/scratch/common_models"

    parser = argparse.ArgumentParser()
    parser.add_argument('--user',help=f'User name, for which a sub-directory must exist')
    parser.add_argument('--data',help = f'Input tsv file')
    parser.add_argument('--llm',help = f'Models:{all_models}')
    parser.add_argument('--merge', action='store_true')
    #parser.set_defaults(merge=False)
    parser.add_argument('--plot', action='store_true', help='Generate KDE plots')
    args = parser.parse_args()

    llm_name = args.llm
    llm_cfg = get_llm_config(llm_name)
    repo_id = llm_cfg['repo_id']
    surp_id = llm_cfg['surp_id']
    bpe_id = llm_cfg['bpe_id']
    ws_ind = llm_cfg['ws_ind']
    char_repl = llm_cfg['char_repl']
    bos_pad = llm_cfg['bos_pad']
    lang = llm_cfg['lang']


    print(f'Model id: {llm_name}')
    device_arg = "auto" if torch.cuda.is_available() else None
    #dtype_arg = torch.float16 if torch.cuda.is_available() else torch.float32
    dtype_arg = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(repo_id, add_prefix_space=True)
    model = AutoModelForCausalLM.from_pretrained(repo_id,device_map=device_arg,dtype=dtype_arg)
    print(f"Model on {model.device}")
    
    df = pd.read_csv(f'{args.user}/data/{args.data}', sep ="\t")
    filename = Path(args.data).stem
    print(f"Data: {filename}")
    long_df = df.apply(surp.process_row, axis=1, args=(model,tokenizer,ws_ind,char_repl, bos_pad, surp_id, lang))
    long_df = pd.concat(long_df.values, ignore_index=True)

    assert len(df) == len(long_df[long_df['is_target']==1]) # collapsed df with target-only surprisals should have same num of rows as original df

    results_path = f'{args.user}/results/llm-surprisal'
    Path(results_path).mkdir(parents=True, exist_ok=True)
    long_df.to_csv(f'{results_path}/{filename}_{llm_name}.tsv', sep='\t', index = False)

    if args.merge:
         surp.merge_surprisal(args.user, args.data)
        
    if args.plot:
         study_cfg = get_study_config(filename)
         params = get_plot_params(study_cfg, filename, llm_name)


         dft = long_df[long_df['is_target']==1] # plot only for target words
         
         plot_out = results_path+"/plots"
         Path(plot_out).mkdir(exist_ok=True)

         surp.kde_plot_conditions(dft, surp_id=surp_id,
                                  outpath=f"{plot_out}/{filename}_{llm_name}_kde.pdf",
                                  xlim=params['xlim'],
                                  ylim=params['ylim'],
                                  title=params['title'],
                                  c_palette=params['c_palette'])