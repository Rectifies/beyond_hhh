# Beyond Helpful, Harmless and Honest

Code and aggregate data for a capability audit testing whether frontier language models produce social engineering content from casual prompts, and whether deployed safety tools detect it.

Five models (GPT-4o, GPT-5.2, Claude Opus 4.6, Claude Sonnet 4.5, Gemini 3 Pro) were tested across 14 categories with 5 runs per prompt-model pair, producing 725 outputs evaluated against three independent safety conditions.

Full write-up: [How Language Models Produce What Safety Tools Can't See](https://rectifies.ai/blog/how-language-models-produce-what-safety-tools-cant-see)

## Repo structure

```
beyond_hhh/scripts         Analysis and evaluation pipeline
  classify_compliance.py   Cross-family compliance classifier
  safety_api_test.py       Three-condition safety evaluation
  collect_outputs.py       Model output collection
  aggregate.py             Compliance aggregation
  config.py                Model and path configuration
beyond_hhh/analysis
  results.md               Aggregate compliance, safety, and factual integrity tables
  figures/                 Charts used in the blog post
beyond_hhh/ 
  requirements.txt         Python dependencies
```

## What's withheld

The prompt set, raw model outputs, and per-output safety evaluations are withheld from this repository for responsible disclosure. Both will be published with the forthcoming paper and are available to researchers on request. 

## Setup

```bash
pip install -r scripts/requirements.txt
```


