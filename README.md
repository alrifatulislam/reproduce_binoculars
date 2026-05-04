# Binoculars — Google Colab Reproduction

This repository contains a Google Colab notebook for reproducing the experiments 
from the Binoculars paper: Spotting LLMs with Binoculars: Zero-Shot Detection of 
Machine-Generated Text (https://arxiv.org/abs/2401.12070).

## What is Binoculars?

Binoculars is a zero-shot, domain-agnostic method for detecting AI-generated text. 
It requires no training data and works by leveraging the overlap in pretraining 
datasets across decoder-only causal language models.

## How This Was Run

1. The Binoculars codebase was uploaded to Google Drive.
2. A Google Colab notebook was created to mount the Drive, install dependencies, 
   and run the experiments end-to-end.
3. All experiments were executed through the notebook using Colab's GPU runtime.

## Notebook

The reproduce.ipynb notebook covers the full pipeline for reproducing the results 
from the paper.

### To Run in Google Colab

1. Upload reproduce.ipynb to Google Colab via File → Upload notebook
2. Mount your Google Drive and place the Binoculars codebase there
3. Set your Hugging Face token as an environment variable:
   export HUGGING_FACE_HUB_TOKEN="your_token_here"
4. Run all cells

## Reference

@misc{hans2024spotting,
      title={Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text}, 
      author={Abhimanyu Hans and Avi Schwarzschild and Valeriia Cherepanova and Hamid Kazemi 
              and Aniruddha Saha and Micah Goldblum and Jonas Geiping and Tom Goldstein},
      year={2024},
      eprint={2401.12070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
