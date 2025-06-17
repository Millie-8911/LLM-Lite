# LLM-Lite
===============================================================
TRANSFORMER-BASED LARGE LANGUAGE MODEL (LLM) FROM SCRATCH
===============================================================

Author: Jou-Chi Huang  
Project Type: Educational / Research (NLP, Deep Learning)  
Language: Python  
Framework: PyTorch  

---------------------------------------------------------------
PROJECT OVERVIEW
---------------------------------------------------------------

This project implements a simplified, lightweight Transformer-based 
Large Language Model (LLM) from scratch to understand the core components 
of modern generative language models. It includes:

- A tokenizer built from scratch
- Multi-head self-attention
- Transformer blocks with residual connections
- Positional encoding
- Training with mixed precision, gradient accumulation, and checkpointing
- Learning rate scheduling (warmup and cosine decay)
- Evaluation: validation loss, perplexity, and attention map visualization

---------------------------------------------------------------
FILE STRUCTURE
---------------------------------------------------------------

.
├── data/                  --> Raw and tokenized text data
├── model/                 --> Transformer model components
├── utils/                 --> Tokenizer, scheduler, visualizations
├── checkpoints/           --> Saved training checkpoints
├── plots/                 --> Output plots (loss, perplexity, LR, attention)
├── train.py               --> Training script
├── eval.py                --> Evaluation and plotting script
├── config.yaml            --> Configurations for model and training
└── README.txt             --> Project description (you’re reading this)

---------------------------------------------------------------
EVALUATION AND VISUALIZATION
---------------------------------------------------------------

After training is complete:

Run:
   python eval.py --checkpoint checkpoints/best_model.pth

Generates:
- Validation loss and perplexity plots
- Learning rate schedule
- Gradient norm statistics
- Attention map visualizations (by layer/head)

All outputs are saved in the /plots/ directory.

---------------------------------------------------------------
SAMPLE CONFIGURATION
---------------------------------------------------------------

- Embedding dimension: 128
- Attention heads: 8
- Transformer layers: 4
- Epochs: 10 and 20
- Optimizer: Adam
- LR Scheduler: Warmup + Cosine decay
- Final validation perplexity: ~1.17

---------------------------------------------------------------
FUTURE WORK
---------------------------------------------------------------

- Fine-tuning with Hugging Face or LoRA
- Domain-specific corpus training (e.g., scientific, math, code)
- Optimizer benchmarking (e.g., Adam vs SGD)
- Ablation studies on attention heads and depth
