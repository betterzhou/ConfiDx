# ConfiDx


## 1. Introduction
This repository contains code for the paper "Uncertainty-Aware Large Language Models for Explainable Disease Diagnosis" (npj Digital Medicine 2025).


## 2. Acknowledgment

This repository is built upon the **LitGPT** framework developed by the Lightning-AI team.  
Most of the training and model implementation code is derived from the official "[LitGPT](https://github.com/Lightning-AI/litgpt)" repository.

We have made some modifications to the original codebase to support the experiments and datasets used in our research paper.  
Full credit for the core model implementation and training framework belongs to the LitGPT authors.


## 3. Usage

### Environment

Install packages:
```bash
pip install 'litgpt[extra]'
```

### Datasets

In this study, we proposed splitting the long instructions into four smaller, more manageable parts. Each part focused on a specific task: disease diagnosis, diagnostic explanation, recognition of diagnostic uncertainty, and explanation of diagnostic uncertainty. To facilitate this, we designed a multi-task learning framework that utilized the four distinct sets of annotated data for instruction fine-tuning.

Generally, each training instance has three components:
1. An instruction describing the task to perform
2. Input data that describes the patient's information or other related information, e.g., clinical note
3. An output that contains the ground-truth, such as diagnosis, diagnostic explanations, or uncertainty label

Please organize the data into JSON format:
```json
{
    "instruction": "You are an experienced doctor. Given a patient's clinical note, please use step-by-step deduction to identify the most likely disease...",
    "input": "A xx-year-old man ...",
    "output": "Diagnosis ..."
}
```

Place the JSON file in the corresponding folder. See "[Tutorial](https://github.com/Lightning-AI/litgpt)" for details.

### Example

Set up the config file before running the code.
```bash
cd ./litgpt/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 litgpt finetune_lora stabilityai/stablelm-base-alpha-3b --device 8 --precision "bf16-true"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 litgpt finetune_lora meta-llama/Meta-Llama-3.1-70B-Instruct --device 8 --precision "bf16-true"
```


## 4. Citation
Please kindly cite the paper if you are interested in our work.
```bib
@article{zhou2025uncertainty,
  title={Uncertainty-aware large language models for explainable disease diagnosis},
  author={Zhou, Shuang and Wang, Jiashuo and Xu, Zidu and Wang, Song and Brauer, David and Welton, Lindsay and Cogan, Jacob and Chung, Yuen-Hei and Tian, Lei and Zhan, Zaifu and others},
  journal={npj Digital Medicine},
  volume={8},
  number={1},
  pages={690},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

or

```
Zhou, S., Wang, J., Xu, Z. et al. Uncertainty-aware large language models for explainable disease diagnosis. npj Digit. Med. 8, 690 (2025). https://doi.org/10.1038/s41746-025-02071-6
```
