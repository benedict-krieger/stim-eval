import argparse
import numpy as np
import pandas as pd
import fasttext
import spacy
from pathlib import Path
import yaml
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_llm_config(llm_name):
    """Load model config"""
    with open("config/llms.yaml", "r") as f:
        llms = yaml.safe_load(f)
    return llms[llm_name]

def get_llm(repo_id):
    """Load Hugging Face tokenizer and model."""
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id)
    model.eval()
    return tokenizer, model

def get_spacy_model(lang):
    """Load spaCy model for stop word removal and lemmatization."""
    with open("config/spacy.yaml", "r") as f:
        config = yaml.safe_load(f)
        model_name = config[lang]
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Error: {model_name} not found. Run: python -m spacy download {model_name}")
        exit(1)

def get_ft_model(lang):
    """Load fastText model"""
    with open("config/fasttext.yaml", "r") as f:
        config = yaml.safe_load(f)
        model_name = config[lang]

    root_dir = Path(os.getenv("FT_MODELS", "."))
    model_path = root_dir / model_name
    return fasttext.load_model(str(model_path))

def cosine_sim(v1, v2):
    """Computes angular distance between two word/sentence vectors."""
    if v1 is None or v2 is None:
        return np.nan
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def get_fasttext_vector(text, nlp, ft_model, use_lemma, use_filter):
    """Pipeline to retrieve sub-word aware vectors from fastText."""
    if pd.isna(text) or str(text).strip() == "":
        return None
        
    doc = nlp(str(text).strip())
    tokens = []
    for token in doc:
        if use_filter and len(doc) > 1 and (token.is_stop or token.is_punct):
            continue
        t = token.lemma_ if use_lemma else token.text
        tokens.append(t)
    
    if not tokens:
        return None

    vectors = [ft_model.get_word_vector(t) for t in tokens]
    return np.mean(vectors, axis=0)

def get_llm_input_vector(text, tokenizer, model, strip_bos=False):
    """Extract and average subword input embeddings from an LLM lookup table."""
    if pd.isna(text) or str(text).strip() == "":
        return None

    inputs = tokenizer(str(text).strip(), add_special_tokens=False, return_tensors="pt")
    input_ids = inputs["input_ids"][0]

    # Defensive Check: Explicitly drop BOS token if tokenizer ignores add_special_tokens=False
    #if strip_bos and len(input_ids) > 0 and input_ids[0] == tokenizer.bos_token_id:
    #    input_ids = input_ids[1:]

    if len(input_ids) == 0:
        return None

    with torch.no_grad():
        wte = model.get_input_embeddings()
        embeddings = wte(input_ids)  # Shape: (num_subwords, embedding_dim)
        mean_embedding = embeddings.mean(dim=0).cpu().numpy()
        
    return mean_embedding

def process_row(row, col1, col2, vector_fn):
    """
    Agile worker function. It doesn't care what model is running; 
    it just executes the vector_fn passed down to it.
    """
    t1, t2 = row[col1], row[col2]
    
    v1 = vector_fn(t1)
    v2 = vector_fn(t2)
    
    return cosine_sim(v1, v2)


##########################################################################################

if __name__ == '__main__':
    llm_list = ['leo13b', 'llammlein7b', 'llammlein1b', 'llammlein120m', 'gerpt2', 'gerpt2large']
    parser = argparse.ArgumentParser(description="Agnostic Semantic Similarity Pipeline")
    parser.add_argument('--user', required=True)
    parser.add_argument('--exp', required=True)
    parser.add_argument('--model', required=True, help=f"fastText:ft, LLM:{llm_list}")
    parser.add_argument('--col1', required=True, help="First column name")
    parser.add_argument('--col2', required=True, help="Second column name")
    parser.add_argument('--lang', required=True, help="German:de, English:en")
    parser.add_argument('--ft_path', default='cc.de.300.bin', help="Path to fastText bin")
    parser.add_argument('--no_lemma', action='store_false', dest='use_lemma', help="Disable lemmatization")
    parser.add_argument('--no_filter', action='store_false', dest='use_filter', help="Disable stopword filter")
    parser.set_defaults(use_lemma=True, use_filter=True)
    args = parser.parse_args()

    print(f"--- Computing {args.model} similarities ---")
    print("Initializing NLP models...")
    nlp = get_spacy_model(args.lang)

    # Define a clean functional wrapper depending on the architecture selected
    if args.model == 'ft':
        print("Loading fastText model...")
        ft_model = get_ft_model(args.lang)
        
        # Package the execution pipeline into a clean single-argument lambda
        vector_fn = lambda text: get_fasttext_vector(
            text, nlp, ft_model, args.use_lemma, args.use_filter
        )
        
    elif args.model in llm_list:
        llm_cfg = get_llm_config(args.model)
        print(f"Loading LLM architecture: {args.model}...")
        
        # Fixed the tuple unpacking bug here (tokenizer, model)
        tokenizer, model = get_llm(llm_cfg["repo_id"])
        
        # Read from your yaml config setup if you have an explicit 'strip_bos' key
        #strip_bos_flag = llm_cfg.get("strip_bos", False)
        
        vector_fn = lambda text: get_llm_input_vector(
            text, tokenizer, model
        )
    else:
        raise ValueError(f"Model identifier '{args.model}' not recognized.")

    # 2. Load Data
    exp_dir = Path(f"users/{args.user}") / args.exp
    input_file = exp_dir / f"{args.exp}.tsv"
    df = pd.read_csv(input_file, sep='\t')

    # 3. Compute Similarity
    col_name = f"{args.col1}_{args.col2}_{args.model}_sim"
    print(f"Computing similarity for: {col_name}")
    
    df[col_name] = df.apply(
        process_row, 
        axis=1, 
        args=(args.col1, args.col2, vector_fn)
    )

    # 4. Export results
    results_dir = exp_dir / "results" / "similarity"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"{args.exp}_{args.model}_sim.tsv"
    
    df.to_csv(out_file, sep='\t', index=False)
    print(f"--- Process Complete ---")
    print(f"Results saved to: {out_file}")