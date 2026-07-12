---
name: nlp
description: This skill should be used when extracting structured information from natural language processing papers. Trigger when the user provides an NLP paper or asks about model types (transformer, RNN), tokenizers, NLP tasks (classification, NER, QA, summarization), benchmarks (GLUE, SQuAD), metrics like BLEU/ROUGE/F1, pretraining data, or parameter counts.
---

# NLP Paper Extraction

## Fields to extract

- **model_type** (required): The type of NLP model (transformer, RNN, etc.)
- **tokenizer**: Tokenizer used (WordPiece, BPE, SentencePiece, etc.)
- **task** (required): The NLP task (classification, NER, QA, summarization, etc.)
- **benchmarks** (required): Benchmarks and datasets used for evaluation
- **metrics** (required): Key reported metrics (BLEU, ROUGE, F1, accuracy, etc.)
- **pretraining_corpus**: Pretraining data source and size
- **parameter_count**: Model size / number of parameters
- **language**: Target language(s), default is English

## Quality checklist for review

- Are the evaluation benchmarks standard for this task?
- Are baseline comparisons fair (same data, similar size)?
- Is statistical significance reported?
- Is the model publicly available (HuggingFace, etc.)?
