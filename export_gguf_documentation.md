
# Exporting a Fine-Tuned Model as a GGUF File

This guide explains the steps to export a fine-tuned model as a GGUF file. It assumes that you have already fine-tuned a model and saved its weights.

## Install Required Packages
```python
!pip install -q -U bitsandbytes
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
```
These commands install necessary packages and libraries. The `-q` flag ensures the installation process is quiet, while `-U` updates the packages to the latest versions.

## Login to Hugging Face Hub

Log into Hugging Face Hub using your token. Replace `"-------"` with your actual token.

```python
from huggingface_hub import login
login(token="-------")
```


## Load the Base Model

Load the model from Hugging Face Hub in bfloat16. 

```python
import torch
import bitsandbytes
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model = "mistralai/Mistral-7B-Instruct-v0.3"
base_model_load = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        return_dict=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
)
```


## Load the Fine-Tuned Weights

Loading the fine-tuned weights using the checkpoint saved in the local drive.

```python
from peft import PeftModel
ft_model = PeftModel.from_pretrained(base_model_load, '/checkpoint475')
```


```python
fine_tuned_model = ft_model.merge_and_unload()
```

## Save the Fine-Tuned Model
```python
fine_tuned_model.save_pretrained("./finetunedmodel_finall")
```
This saves the fine-tuned model locally in the finetunedmodel_finall directory.

## Save the Tokenizer 
```python
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained("./finetunedmodel_finall")
```
This code saves the tokenizer associated with the model.

## Clone the llama.cpp Repository
```python
!git clone https://github.com/ggerganov/llama.cpp.git
```
This clones the `llama.cpp` repository, which contains tools needed for converting the model.

## Navigate to the Cloned Repository
```python
%cd /content/llama.cpp
```
This changes the current working directory to the cloned repository.

## Install Required Build Tools
```python
!apt update -y
!apt install build-essential git cmake libopenblas-dev libeigen3-dev -y
```
These commands update the package list and install necessary build tools.

## Build the llama.cpp Project
```python
!make LLAMA_OPENBLAS=1
```
This builds the `llama.cpp` project with OpenBLAS support.

## Install Python Requirements
```python
!python3 -m pip install -r requirements.txt
```
This installs the Python dependencies required by `llama.cpp`.

## Convert the Model to GGUF Format
```python
!python convert_hf_to_gguf.py finetunedmodel_finall/
```
This converts the fine-tuned model to the GGUF format using the `convert-hf-to-gguf.py` script.

