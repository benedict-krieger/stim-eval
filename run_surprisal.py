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

def get_exp_config(expname):
    """
    Attempts to find plot settings based on the input expname.
    """
    with open("config/exp.yaml", "r") as f:
            all_configs = yaml.safe_load(f)
    
    # Look for the expname (e.g., 'test' from 'test.tsv')
    return all_configs.get(expname, {})

def get_plot_params(exp_cfg, expname, llm_name):
    """
    Applies fallback logic using exp_cfg dict.
    """
    return {
        'xlim': exp_cfg.get('x_lim', None),
        'ylim': exp_cfg.get('y_lim', None),
        'title': exp_cfg.get('title', f"{expname} {llm_name} surprisal"),
        'c_palette': exp_cfg.get('c_palette', "husl"),
        'c_labels': exp_cfg.get('c_labels', None),
        'show_legend': exp_cfg.get('show_legend', True)
    }


if __name__ == '__main__':

    all_models = ['leo13b', 'llammlein7b', 'llammlein1b', 'llammlein120m', 'gerpt2', 'gerpt2-large']
    #base_model_path = "/scratch/common_models"

    parser = argparse.ArgumentParser()
    parser.add_argument('--user',help=f'User name, for which a sub-directory must exist')
    parser.add_argument('--exp',help = f'Name of experiment and input tsv file')
    parser.add_argument('--llm',help = f'Models:{all_models}')
    parser.add_argument('--plot', action='store_true', help='Generate KDE plots')
    args = parser.parse_args()

    llm_cfg = get_llm_config(args.llm)
    repo_id = llm_cfg['repo_id']
    surp_id = llm_cfg['surp_id']
    bpe_id = llm_cfg['bpe_id']
    ws_ind = llm_cfg['ws_ind']
    char_repl = llm_cfg['char_repl']
    bos_pad = llm_cfg['bos_pad']
    lang = llm_cfg['lang']


    print(f'Model id: {args.llm}')
    device_arg = "auto" if torch.cuda.is_available() else None
    #dtype_arg = torch.float16 if torch.cuda.is_available() else torch.float32
    dtype_arg = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(repo_id, add_prefix_space=True)
    model = AutoModelForCausalLM.from_pretrained(repo_id,device_map=device_arg,dtype=dtype_arg)
    print(f"Model on {model.device}")


    print(f"Data: {args.exp}")
    exp_dir = Path(f"users/{args.user}") / args.exp
    input_file = exp_dir / f"{args.exp}.tsv"
    df = pd.read_csv(input_file, sep ="\t")
    long_df = df.apply(surp.process_row, axis=1, args=(model,tokenizer,ws_ind,char_repl, bos_pad, surp_id, lang))
    long_df = pd.concat(long_df.values, ignore_index=True)

    #assert len(df) == len(long_df[long_df['is_target']==True]) # collapsed df with target-only surprisals should have same num of rows as original df
    if len(df) != len(long_df[long_df['is_target']==True]): # collapsed df with target-only surprisals should have same num of rows as original df
         print("Warning: shape mismatch between original data and collapsed surprisal data detected.")

    results_path = exp_dir / "results" / "llm-surprisal"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    long_df.to_csv( results_path / f'{args.exp}_{args.llm}.tsv', sep='\t', index = False)
    
    if args.plot:
         exp_cfg = get_exp_config(args.exp)
         params = get_plot_params(exp_cfg, args.exp, args.llm)


         dft = long_df[long_df['is_target']==True] # plot only for target words
         
         plot_out = results_path / "plots"
         Path(plot_out).mkdir(exist_ok=True)

         if params['c_labels']:
            # Map the 'Condition' column names from config
            dft = dft.copy() # Avoid SettingWithCopyWarning
            dft['Condition'] = dft['Condition'].map(params['c_labels']).fillna(dft['Condition'])

         surp.kde_plot_conditions(dft, surp_id=surp_id,
                                  outpath=plot_out / f"{args.exp}_{args.llm}_kde.pdf",
                                  xlim=params['xlim'],
                                  ylim=params['ylim'],
                                  title=params['title'],
                                  c_palette=params['c_palette'],
                                  show_legend=params['show_legend'])