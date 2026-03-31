# stim-eval

A collection of tools to evaluate stimulus materials for psycholinguistic experiments.

- LLM surprisal
- Word frequencies
- Semantic relatedness (fastText)

Planned branches:

-  Local (main)
- Tortoise CPU server
- Coli GPU cluster

Takes tab-separated CSV files as input, which have one stimulus per line.

Mandatory columns:

- "Stimulus": the stimulus sentence including the target word
- "Target": the target word


Output columns:

- "\[llm_name\]_surp": surprisal estimates from the specified large language model
