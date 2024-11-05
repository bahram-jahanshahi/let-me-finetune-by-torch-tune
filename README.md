# Fine Tune Llama 3.2 3b by Torch Tune
https://github.com/pytorch/torchtune 

### Might be better to fine tune Llama 3.2 1b
### Might be better to use QLoRa 
### also check wich Finetuning needs which device on the GitHub page https://github.com/pytorch/torchtune 

## Install TorchTune
Visit this page https://pytorch.org/torchtune/stable/install.html 
```shell
pip install torch torchvision torchao
pip install torchtune
```
### Installation confirm
Run this command
```shell
tune
```

## Finetune
Visit this page https://pytorch.org/torchtune/stable/tutorials/first_finetune_tutorial.html  

### Downloading a model
```shell
tune download meta-llama/Llama-3.2-3B-Instruct --output-dir tmp/Llama-3.2-3B-Instruct --hf-token hf_...
```
The model will be downloaded in `tmp`folder.  
At this moment this does not download 
```shell
checkpoint_files: [
    model-00001-of-00002.safetensors,
    model-00002-of-00002.safetensors,
  ]
```
so it's needed to visit the model card and download them manually
and put it in the directory https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/tree/main 

### Selecting Recipe
```shell
tune ls
```
For meta-llama/Llama-3.2-3B we need to choose llama3_2/3B_lora_single_device recipe

### Customizing the config
```shell
tune cp llama3_2/3B_lora_single_device custom_config.yaml
```
Open the file `custom_config.yaml` and change all the `/tmp` to `tmp` because we want to download the models in this project  
I also change the device to mps
```yaml
# Environment
# device: cuda
device: mps
dtype: bf16
```
I also changed the dtype
```yaml
#dtype: bf16
dtype: fp32
```
Also changed the lora rank
```yaml
lora_rank: 8 # 64
```

### Run
```shell
tune run lora_finetune_single_device --config custom_config.yaml epochs=1
```
or 
```shell
tune run lora_finetune_single_device --config custom_dataset_config.yaml epochs=1
```  
## How to upload the huge LLM files to a server via SSH
```shell
scp -i /<HOME_DIR>/.ssh/id_sha -P <PORT_NUMBER> <MODEL_FILE_PATH> <USERNAME>@<SERVER_IP>:/workspace
```

## Datasets
Download `alpaca_data_cleaned.json` file here on huggingface
https://huggingface.co/datasets/yahma/alpaca-cleaned/tree/main  
It's a big file with 50_000 records. for testing the finetune process it's needed to reduce it. I deleted 80% of the prompts. 









