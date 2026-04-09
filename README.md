# Selective RAG: Training RAG via Evidence-Aware Selective Answering

This repository contains the code for the project, **"정답 근거 유무에 따른 RAG 모델의 선택적 응답 학습 (Training RAG via Evidence-Aware Selective Answering)"**, which is being prepared for submission to the Korea Computer Congress (KCC) 2026. 

This research was conducted as part of a winter research internship at **ETRI (Electronics and Telecommunications Research Institute)** from January 2026 to March 2026.

## 📌 Overview
Retrieval-Augmented Generation (RAG) systems heavily rely on retrieved documents. However, in real-world scenarios, retrievers often fail to provide documents containing the evidence needed to answer a query, leading to model **hallucination**. 

To address this, we propose **Selective RAG**, a framework that trains the model to distinguish between situations where it should generate an answer and where it should abstain (reject) based on the presence of evidence in the provided documents. Our method significantly reduces hallucinations while maintaining high precision, improving Overall Accuracy by up to 10%p in noisy retrieval environments.

## 📂 Repository Structure
```text
.
├── configs/                  # Configuration files for training and models
│   ├── ds_zero2_fp16.json    # DeepSpeed Zero-2 configuration
│   ├── proposed.yaml
│   ├── qwen25_1p5b.yaml      # Config for 1.5B model
│   └── qwen25_7b.yaml        # Config for 7B model
├── data/                     # Sample datasets and format references
├── scripts/                  # Shell scripts for running experiments
│   ├── env.sh                # Environment variables setup
│   ├── run_eval_cuda0_1p5b.sh # Evaluation script for 1.5B model
│   ├── run_eval_cuda1_7b.sh   # Evaluation script for 7B model
│   ├── run_sweep.sh          # Scripts for hyperparameter/p_train sweeping
│   ├── run_sweep_2.sh
│   └── run_sweep_3.sh
└── src/                      # Source code
    ├── dataset_sft.py        # Dataset processing for Supervised Fine-Tuning
    ├── eval.py               # Evaluation code
    └── train.py              # Training code (LoRA-based PEFT)
```

## 📊 Dataset Notice
The original dataset used in this research was provided by ETRI and is **not publicly available** in this repository due to strict copyright and security policies.

However, if you wish to run this code with your own data, please structure your dataset following the example format. **Please refer to the sample dataset files uploaded in the `data/` folder** to understand the exact input structure required for training and evaluation (e.g., query, contexts, answer, and evidence indicators).

## 💡 Usage & Code Reference
This repository serves as an archive for the research code. While a plug-and-play execution is not possible without the original dataset, you can refer to the scripts below to understand our experimental setup and training flow. 

All experiments in the paper were primarily conducted using Qwen models via LoRA-based Parameter-Efficient Finetuning (PEFT).

* **Environment Setup:** `source scripts/env.sh`
* **Training & Sweeps:** Check `scripts/run_sweep.sh` to see how we varied the probability of evidence presence ($p_{train}$) during training.
* **Evaluation:** Run `scripts/run_eval_cuda0_1p5b.sh` or `scripts/run_eval_cuda1_7b.sh` to evaluate the model's Overall Accuracy, Precision, and Hallucination rates across answerable and unanswerable scenarios.

## 📜 Citation
Since the paper is currently under preparation, if you find this repository helpful for your research, please consider citing it as follows:
```bibtex
@misc{park2026selective_repo,
  author = {Park, Jungwon and others},
  title = {Training RAG via Evidence-Aware Selective Answering},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/Jungwonpark1027/](https://github.com/Jungwonpark1027/)<repository-name>}}
}
```
