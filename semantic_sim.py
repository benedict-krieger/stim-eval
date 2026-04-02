import argparse
import numpy as np
import pandas as pd
import fasttext
import spacy
from pathlib import Path

# --- GLOBAL UTILITIES ---

def get_spacy_model(model_name="de_core_news_lg"):
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Error: {model_name} not found. Run: python -m spacy download {model_name}")
        exit(1)

def cosine_sim(v1, v2):
    """Computes the cosine similarity between two vectors."""
    if v1 is None or v2 is None:
        return np.nan
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

# --- CORE LOGIC (The "Worker" Functions) ---

def get_custom_vector(text, nlp, ft_model, use_lemma, use_filter):
    """Linguistic pipeline: spaCy Cleaning -> FastText Vectors."""
    if pd.isna(text) or str(text).strip() == "":
        return None
        
    doc = nlp(str(text))
    
    tokens = []
    for token in doc:
        if use_filter and (token.is_stop or token.is_punct):
            continue
        # We use lemma or text based on user preference (No .lower() as requested)
        t = token.lemma_ if use_lemma else token.text
        tokens.append(t)
    
    if not tokens:
        return None

    # Retrieve vectors from FastText and average them
    vectors = [ft_model.get_word_vector(t) for t in tokens]
    return np.mean(vectors, axis=0)

def process_row(row, nlp, ft_model, col1, col2, method, use_lemma, use_filter):
    """Processes a single row and returns a Series of similarity scores."""
    t1, t2 = str(row[col1]), str(row[col2])
    results = {}

    # Method 1: FastText
    if method in ['fasttext', 'both']:
        v1 = get_custom_vector(t1, nlp, ft_model, use_lemma, use_filter)
        v2 = get_custom_vector(t2, nlp, ft_model, use_lemma, use_filter)
        results['ft_sim'] = cosine_sim(v1, v2)

    # Method 2: spaCy Internal
    if method in ['spacy', 'both']:
        # Note: .similarity() on docs uses averaged vectors internally
        results['spacy_sim'] = nlp(t1).similarity(nlp(t2))
            
    return pd.Series(results)

# --- SCRIPT EXECUTION (The "Orchestrator") ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Semantic Similarity Pipeline")
    parser.add_argument('--user', required=True)
    parser.add_argument('--exp', required=True)
    parser.add_argument('--col1', required=True)
    parser.add_argument('--col2', required=True)
    parser.add_argument('--method', choices=['fasttext', 'spacy', 'both'], default='both')
    parser.add_argument('--ft_path', default='cc.de.300.bin')
    parser.add_argument('--no_lemma', action='store_false', dest='use_lemma')
    parser.add_argument('--no_filter', action='store_false', dest='use_filter')
    parser.set_defaults(use_lemma=True, use_filter=True)
    args = parser.parse_args()

    # 1. Setup Models
    nlp = get_spacy_model("de_core_news_lg")
    ft = None
    if args.method in ['fasttext', 'both']:
        print("Loading FastText...")
        ft = fasttext.load_model(args.ft_path)

    # 2. Setup Data
    exp_dir = Path(args.user) / args.exp
    input_path = exp_dir / f"{args.exp}.tsv"
    df = pd.read_csv(input_path, sep='\t')

    # 3. Compute (The clean apply call)
    print(f"Processing similarities for {args.exp}...")
    sim_df = df.apply(
        process_row, 
        axis=1, 
        args=(nlp, ft, args.col1, args.col2, args.method, args.use_lemma, args.use_filter)
    )
    
    # Combine original data with new similarity columns
    final_df = pd.concat([df, sim_df], axis=1)

    # 4. Save
    out_dir = exp_dir / "results" / "similarity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.exp}_{args.method}_sim.tsv"
    
    final_df.to_csv(out_file, sep='\t', index=False)
    print(f"Success! Results saved to: {out_file}")