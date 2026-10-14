import argparse
import traceback
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import surprisal as surp

# Paths are built from this file's location (src/run_surprisal.py), not from the folder
# the job happens to be started in. That makes the script work no matter where HTCondor launchs it from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # src/ -> project root
CONFIG_DIR = PROJECT_ROOT / "config"
USERS_DIR = PROJECT_ROOT / "users"


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_exp_config(expname):
    """Plot settings for one experiment (empty dict if none are configured)."""
    return load_yaml(CONFIG_DIR / "exp.yaml").get(expname, {})


def get_plot_params(exp_cfg, expname, llm_name):
    """Applies fallback logic using exp_cfg dict."""
    return {
        'xlim': exp_cfg.get('x_lim', None),
        'ylim': exp_cfg.get('y_lim', None),
        'title': exp_cfg.get('title', f"{expname} {llm_name} surprisal"),
        'c_palette': exp_cfg.get('c_palette', "husl"),
        'c_labels': exp_cfg.get('c_labels', None),
        'show_legend': exp_cfg.get('show_legend', True)
    }


def resolve_model_path(llm_cfg, model_dir):
    """
    Use the local copy <model_dir>/<local_dir> if it exists,
    otherwise fall back to the Hugging Face repo id.
    """
    local_dir = llm_cfg.get('local_dir')
    if model_dir and local_dir:
        local_path = Path(model_dir) / local_dir
        if local_path.is_dir():
            return str(local_path)
        print(f"Warning: {local_path} not found, falling back to {llm_cfg['repo_id']}")
    return llm_cfg['repo_id']


def run_experiment(exp, args, llm_cfg, model, tokenizer):
    print(f"\n=== Experiment: {exp} ===")
    exp_dir = USERS_DIR / args.user / exp
    df = pd.read_csv(exp_dir / f"{exp}.tsv", sep="\t")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{exp}"):
        rows.append(surp.process_row(row, model, tokenizer,
                                     llm_cfg['ws_ind'], llm_cfg['char_repl'], llm_cfg['bos_pad'],
                                     llm_cfg['surp_id'], llm_cfg['lang']))
    long_df = pd.concat(rows, ignore_index=True)

    # collapsed df with target-only surprisals should have same num of rows as original df
    if len(df) != long_df['is_target'].sum():
        print("Warning: shape mismatch between original data and collapsed surprisal data detected.")

    results_path = exp_dir / "results" / "llm-surprisal"
    results_path.mkdir(parents=True, exist_ok=True)
    out_file = results_path / f'{exp}_{args.llm}.tsv'
    long_df.to_csv(out_file, sep='\t', index=False)

    if args.plot:
        print("Producing density plots")
        params = get_plot_params(get_exp_config(exp), exp, args.llm)
        dft = long_df[long_df['is_target']].copy()  # plot only for target words
        plot_out = results_path / "plots"
        plot_out.mkdir(exist_ok=True)
        if params['c_labels']:
            dft['Condition'] = dft['Condition'].map(params['c_labels']).fillna(dft['Condition'])
        surp.kde_plot_conditions(dft, surp_id=llm_cfg['surp_id'],
                                 outpath=plot_out / f"{exp}_{args.llm}_kde.pdf",
                                 xlim=params['xlim'], ylim=params['ylim'],
                                 title=params['title'], c_palette=params['c_palette'],
                                 show_legend=params['show_legend'])

    print(f"Results saved to: {out_file}")


if __name__ == '__main__':

    all_llms = load_yaml(CONFIG_DIR / "llms.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument('--user', required=True, help='User name, for which a sub-directory must exist in users/')
    parser.add_argument('--exp', required=True, nargs='+',
                        help='One or more experiment names (= input tsv names). The model is loaded once for all.')
    parser.add_argument('--llm', required=True, choices=list(all_llms))
    parser.add_argument('--model_dir', default=None,
                        help='Folder with local model copies, e.g. /local/common_models. '
                             'If omitted or not found, the model is loaded via its repo_id.')
    parser.add_argument('--plot', action='store_true', help='Generate KDE plots')
    args = parser.parse_args()

    llm_cfg = all_llms[args.llm]
    model_path = resolve_model_path(llm_cfg, args.model_dir)
    dtype = getattr(torch, llm_cfg.get('dtype', 'float32'))  # 'bfloat16' -> torch.bfloat16

    print("--- Computing LLM surprisal ---")
    print(f"Model id:   {args.llm}")
    print(f"Loading:    {model_path}")
    print(f"dtype:      {dtype}")
    print(f"GPUs seen:  {torch.cuda.device_count()}")

    device_arg = "auto" if torch.cuda.is_available() else None
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_arg, dtype=dtype)
    model.eval()
    print(f"Model on {model.device}")
    if hasattr(model, "hf_device_map"):
        used = sorted(set(str(d) for d in model.hf_device_map.values()))
        print(f"Model layers spread over: {used}")

    # Load model once, process several experiments.
    # If one experiment fails, the others still run.
    failed = []
    for exp in args.exp:
        try:
            run_experiment(exp, args, llm_cfg, model, tokenizer)
        except Exception:
            traceback.print_exc()
            failed.append(exp)

    print("--- Process Complete ---")
    if failed:
        print(f"Failed experiments: {failed}")
        raise SystemExit(1)
