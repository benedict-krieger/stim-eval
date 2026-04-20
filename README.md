# stim-eval

A collection of tools to evaluate stimulus materials for psycholinguistic experiments.

- LLM surprisal
- Word frequencies
- Semantic relatedness (fastText)

Planned branches:

- CPU server
- GPU cluster

## Required Python packages
Pandas
Numpy
Pytorch
Transformers
Matplotlib
Seaborn
Wordfreq
Fasttext
Spacy

## Input

Create a folder with your `<USERNAME>` within the `users` directory, and an experiment folder with `<EXPNAME>` within this user folder.
Place a tab-separated (.tsv) file (also named as `<EXPNAME>`) in the experiment folder.
Input tsv files should have one stimulus per line and the following columns:

- "Stimulus": the stimulus sentence including the target word
- "Target": the target word, must match the target word in the stimulus column
- "ItemNum"
- "Condition"


## LLM surprisal

Estimates word-by-word surprisal for the stimuli, transforming the original data into a long format.

```python run_surprisal.py --user <USERNAME> --exp <EXPNAME> --llm <LLMNAME>```

The `USERNAME` and `EXPNAME` arguments should match what you have created above.
Currently, for `LLMNAME` a number of German LLMs are supported, which are listed in config/llms.yaml.

Optional flags:

- `plot`: create density plots for the target word surprisals per condition
- `merge`: scan results for multiple files from different LLMs and merge them into a single file

Output columns:

- `LLMNAME_surp`: surprisal estimates from the specified large language model
- `word`: untokenized word, surprisal have been summed in case of sub-word splits
- `word_position`: position of the word in the stimulus sentence
- `word_freq`: Zipf frequency computed with the Wordfreq library
- `word_length`: number of characters
- `is_target`: TRUE for the target word. If the target appears multiple times, only the last occurrence is flagged as TRUE.


## Semantic relatedness

Computes cosine similarity between the fastText embeddings of the words/sentences of two columns of the input data. If a given string has more than one word, an average representation across individual words is computed. Words are lemmatized and stopwords are removed using spaCy.

```python semantic_sim.py --user <USERNAME> --exp <EXPNAME> --col1 <COL1NAME> --col2 <COL2NAME>```

Optional flags:

- `no_lemma`: disable lemmatization
- `no_filter`: disable stopword filtering