import argparse
import numpy as np
import pandas as pd
import fasttext
import spacy
from pathlib import Path
import yaml
import os


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
    """
    Pipeline: 
    1. (Optionally) transform text with spaCy (Stopword/punctuation filtering and lemmatization).
    2. Retrieve sub-word aware vectors from fastText.
    """
    if pd.isna(text) or str(text).strip() == "":
        return None
        
    # Process text, keep casing for spaCy's POS tagger
    doc = nlp(str(text).strip())
    
    tokens = []
    for token in doc:
        # Optional filtering: stopwords and punctuation
        if use_filter and (token.is_stop or token.is_punct):
            continue
        # Optional lemmatization
        t = token.lemma_ if use_lemma else token.text
        tokens.append(t)
    
    if not tokens:
        return None

    # Compute centroid of token vectors
    vectors = [ft_model.get_word_vector(t) for t in tokens]
    return np.mean(vectors, axis=0)

def process_row(row, nlp, ft_model, col1, col2, use_lemma, use_filter):
    """Worker function for pandas apply to compute similarity per row."""
    t1, t2 = row[col1], row[col2]
    
    v1 = get_fasttext_vector(t1, nlp, ft_model, use_lemma, use_filter)
    v2 = get_fasttext_vector(t2, nlp, ft_model, use_lemma, use_filter)
    
    return cosine_sim(v1, v2)


##########################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="fastText Semantic Similarity")
    parser.add_argument('--user', required=True)
    parser.add_argument('--exp', required=True)
    parser.add_argument('--col1', required=True, help="First column name")
    parser.add_argument('--col2', required=True, help="Second column name")
    parser.add_argument('--lang', required=True, help="German:de, English:en")
    parser.add_argument('--ft_path', default='cc.de.300.bin', help="Path to fastText bin")
    parser.add_argument('--no_lemma', action='store_false', dest='use_lemma', help="Disable lemmatization")
    parser.add_argument('--no_filter', action='store_false', dest='use_filter', help="Disable stopword filter")
    parser.set_defaults(use_lemma=True, use_filter=True)
    args = parser.parse_args()

    print("--- Computing fastText similarities ---")
    # 1. Initialize
    print("Initializing NLP models...")
    nlp = get_spacy_model(args.lang)
    ft = get_ft_model(args.lang)

    # 2. Load Data
    exp_dir = Path(f"users/{args.user}") / args.exp
    input_file = exp_dir / f"{args.exp}.tsv"
    df = pd.read_csv(input_file, sep='\t')

    # 3. Compute Similarity
    col_name = f"{args.col1}_{args.col2}_ft_sim"
    print(f"Computing similarity for: {col_name}")
    
    df[col_name] = df.apply(
        process_row, 
        axis=1, 
        args=(nlp, ft, args.col1, args.col2, args.use_lemma, args.use_filter)
    )

    # 4. Export results
    results_dir = exp_dir / "results" / "similarity"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"{args.exp}_ft_sim.tsv"
    
    df.to_csv(out_file, sep='\t', index=False)
    print(f"--- Process Complete ---")
    print(f"Results saved to: {out_file}")