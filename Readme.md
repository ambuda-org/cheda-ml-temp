# Sanskrit Segmentation Using Transformers

Starting with just compound splitting and removing sandhi

## Status

A very simple char seq2seq transformer model is tested for Sanskrit Segmentation (removing sandhi only). More work needs to be done.

## Install Required Packages

If you have a GPU:

```
virtualenv venv
source venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install tqdm wandb pandas BeautifulSoup4 lxml
```

CPU Only:

```
virtualenv venv
source venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install tqdm wandb pandas BeautifulSoup4 lxml
```

## Prepare Dataset

```bash
chmod +x fetch_data.sh
./fetch_data.sh
python3 prepare_dataset.py
```

## Train Model

```bash
python3 train.py
```

### Note:

Some code was taken from the following repository. See License/ {MIT License}

https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/seq2seq_transformer
