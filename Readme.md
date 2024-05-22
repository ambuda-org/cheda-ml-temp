# Sanskrit Segmentation Using Transformers

Starting with just compound splitting and removing sandhi

## Install Required Packages

1. numpy
2. torch
3. wandb
4. tqdm
5. pandas
6. lxml

If you have a GPU:

```
virtualenv venv
source venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install tqdm wandb pandas BeautifulSoup lxml
chmod +x fetch_data.sh
./fetch_data.sh
python3 prepare_dataset.py
```

CPU Only:

```
virtualenv venv
source venv/bin/activate
pip3 install requirements_cpu.txt
chmod +x fetch_data.sh
./fetch_data.sh
python3 prepare_dataset.py
```
