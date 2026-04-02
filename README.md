# stim-eval

A collection of tools to evaluate stimulus materials for psycholinguistic experiments.

- LLM surprisal
- Word frequencies
- Semantic relatedness (fastText)

Planned branches:

- Local (main)
- Tortoise CPU server
- Coli GPU cluster

Takes tab-separated CSV files as input, which have one stimulus per line.

Mandatory columns:

- "Stimulus": the stimulus sentence including the target word
- "Target": the target word, must match the target word in the stimulus column
- "ItemNum"
- "Condition"


## LLM surprisal

Output columns:

- "\[llm_name\]_surp": surprisal estimates from the specified large language model
- word
- word_position
- word_freq
- word_length
- is_target


## Semantic relatedness
