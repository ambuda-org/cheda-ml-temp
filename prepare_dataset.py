import re
from bs4 import BeautifulSoup
import subprocess
import os
import sys
import numpy as np
import random
from tokenizer import CharTokenizer

############################################################################################################################
############################################################################################################################
def line():
  print("==============================================================================================================================")

UNK_TOKEN_ID = 0
PAD_TOKEN_ID = 1
SOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3

############################################################################################################################
############################################################################################################################

# Mahabharata, Shatakatraya need some more work
input_files = [
    ("data/dcs/amarushatakam.txt", "data/texts/gretil/sa_amaru-amaruzataka.xml"),
    ("data/dcs/hamsadutam.txt", "data/texts/gretil/sa_rUpagosvAmin-haMsadUta.xml"),
    ("data/dcs/kumarasambhavam.txt", "data/texts/gretil/sa_kAlidAsa-kumArasaMbhava.xml"),
    ("data/dcs/mukundamala.txt", "data/texts/gretil/sa_kulazekhara-mukundamAlA-eddurgaprasad.xml"),
    ("data/dcs/rtusamharam.txt", "data/texts/gretil/sa_kAlidAsa-RtusaMhAra.xml"),
    ("data/dcs/bodhicaryavatara.txt", "data/texts/gretil/sa_zAntideva-bodhicaryAvatAra.xml"),
    ("data/dcs/kiratarjuniyam.txt", "data/texts/gretil/sa_bhAravi-kirAtArjunIya.xml"),
    # ("data/dcs/mahabharatam.txt", None),
    ("data/dcs/ramayanam.txt", "data/texts/gretil/sa_rAmAyaNa.xml"),
    ("data/dcs/saundaranandam.txt", "data/texts/gretil/sa_azvaghoSa-saundarAnanda-edmatsunami.xml"),
    ("data/dcs/caurapancashika.txt", "data/texts/gretil/sa_bilhaNa-caurapaJcAzikA.xml"),
    ("data/dcs/kokilasandesha.txt", "data/texts/gretil/sa_uddaNDa-kokilasaMdesa.xml"),
    ("data/dcs/meghadutam-kale.txt", "data/texts/gretil/sa_kAlidAsa-meghadUta-edkale.xml")
    # ("data/dcs/shatakatrayam.txt", "data/texts/gretil/sa_bhatRhari-zatakatraya.xml")
]


def getItem(id, isKira, isBodhi, isKoki, isKu, isMegh, isRitu, isRam):
    if isRam:
        parts = id.split('.')
        major = int(parts[1])
        minor = int(parts[2])
        patch = int(parts[3])
        
        new_id = f"R_{major}.{minor:03d}.{patch:03d}"
        return new_id
    
    item = "Bca" if isBodhi else "BhKir" if isKira else "Kok" if isKoki else "Ks" if isKu else "KalMgD" if isMegh else "KalRs" if isRitu else None
    
    if item is None:
        return "NaN"

    for it in id.split('.')[1:]:
        item += '.'
        item += it
    return item

def extract_ids_and_entries(txt_path):
    content = None
    with open(txt_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    ids = re.findall(r'# id = (.+)', content)
    entries = content.split("# id = ")[1:]
    
    entries_dict = {entry.split('\n', 1)[0] : entry.split('\n', 1)[1] for entry in entries}
    
    # Edge Case (too many to be called edge :clown: )
    isKira = ids[0].startswith("Kira")
    isBodhi = ids[0].startswith("BoCa")
    isKoki = ids[0].startswith("Kokila")
    isKu = ids[0].startswith("Ku")
    isMegh = ids[0].startswith("Megh")
    isRitu = ids[0].startswith("Rtu")
    isRam = ids[0].startswith("R.")

    if isKira or isBodhi or isKoki or isKu or isMegh or isRitu or isRam:
        _ids = []
        _entries_dict = {}
        for id in ids:
            item = getItem(id, isKira, isBodhi, isKoki, isKu, isMegh, isRitu, isRam)
            _ids.append(item)
        
        for key, value in entries_dict.items():
            item = getItem(key, isKira, isBodhi, isKoki, isKu, isMegh, isRitu, isRam)
            _entries_dict[item] = value
        
        ids = _ids
        entries_dict = _entries_dict
    
    return ids, entries_dict

def extract_orig_text(xml_path, ids):
    original_texts = {}
    ids_set = set(ids)
    with open(xml_path, 'r', encoding='utf-8') as tei:
        soup = BeautifulSoup(tei, features="xml")
        tags = soup.find_all('lg')
        for t in tags:
            tag_id = t.get("xml:id")
            if tag_id in ids_set:
                text = ''.join(t.stripped_strings)
                original_texts[tag_id] = text
                result = subprocess.run(
                    ['./vidyut/target/release/lipi', '--from', 'iast', '--to', 'slp1', text],
                    capture_output=True,
                    text=True
                )
                original_texts[tag_id] = ""
                text = result.stdout
                for i in text.split('|'):
                    i = i.strip()
                    if len(i) > 0 and not i[0].isdigit() and not i[0] == ')' and not i[0] == '(':
                        original_texts[tag_id] += (i.strip())
    return original_texts

def combine_and_write(txt_path, xml_path, output_path):
    ids, entries_dict = extract_ids_and_entries(txt_path)
    original_texts = extract_orig_text(xml_path, ids)
    with open(output_path, 'w', encoding='utf-8') as file:
        for id in ids:
            file.write(f'# id = {id}\n')
            if id in original_texts:
                file.write(
                    re.sub(r'//$', '', original_texts[id].strip())
                    )
                file.write("\n")
            file.write(entries_dict[id])

############################################################################################################################
############################################################################################################################

for directory in ["data/dcs", "data/texts"]:
    if not os.path.exists(directory):
        line()
        print(f"Error: Directory '{directory}' not found.")
        print("Error: Dataset not initialized!")
        print("Run `fetch_data.sh` !!!")
        line()
        sys.exit(1)

if not os.path.exists("data/dataset"):
    os.mkdir("data/dataset")
    line()
    for txt_path, xml_path in input_files:
        print(txt_path)
        if not xml_path:
            continue
        output_path = "data/dataset/" + txt_path.split('/')[-1]
        combine_and_write(txt_path, xml_path, output_path)
        line()

############################################################################################################################
############################################################################################################################

# Using only the sandhi splits for now

tokenizer = CharTokenizer()

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    sections = content.split("# id =")
    dataset = []
    max_len_original = 0
    max_len_target = 0

    for section in sections:
        if section.strip():
            lines = section.strip().split('\n')
            original_line = '_'.join(lines[1].strip().split(' '))
            target_lines = '_'.join([line.split('\t')[0].strip() for line in lines[2:] if line.strip()])
            original_enc = np.array(tokenizer.encode(original_line))
            target_enc = np.array(tokenizer.encode(target_lines))
            max_len_original = max(max_len_original, original_enc.shape[0])
            max_len_target = max(max_len_target, target_enc.shape[0])
            dataset.append([original_enc, target_enc])

    return dataset, max_len_original, max_len_target

def pad_sequence(sequence, max_length):
    padded_sequence = np.full((max_length, 1), PAD_TOKEN_ID, dtype=int)
    padded_sequence[:sequence.shape[0], 0] = sequence.flatten()
    return padded_sequence



def create_dataset(directory):
    dataset = []
    max_len_original = 0
    max_len_target = 0

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            enc, max_orig, max_tgt = process_file(file_path)
            max_len_original = max(max_len_original, max_orig)
            max_len_target = max(max_len_target, max_tgt)
            for e in enc:
                dataset.append(e)

    padded_dataset = []
    for original_enc, target_enc in dataset:
        padded_orig = pad_sequence(original_enc, max_len_original)
        padded_tgt = pad_sequence(target_enc, max_len_target)
        padded_dataset.append([padded_orig, padded_tgt])

    inputs = []
    targets = []
    for inp, tgt in padded_dataset:
        inputs.append(inp)
        targets.append(tgt)

    inputs = np.array(inputs)
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1])

    targets = np.array(targets)
    targets = targets.reshape(targets.shape[0], targets.shape[1])
    
    return inputs, targets

def save_dataset(dataset, output_file):
    np.save(output_file, dataset)

input_directory = 'data/dataset'

inputs, targets = create_dataset(input_directory)


############################################################################################################################
############################################################################################################################


dataset_size = len(inputs)
test_size = 0.2  # 20% of the dataset for testing

dataset = list(zip(inputs, targets))
random.shuffle(dataset)

test_split_index = int(dataset_size * (1 - test_size))
train_set = dataset[:test_split_index]
test_set = dataset[test_split_index:]

inputs_train, targets_train = zip(*train_set)
inputs_test, targets_test = zip(*test_set)

save_dataset(inputs_train, 'train_inputs.npy')
save_dataset(targets_train, 'train_targets.npy')

save_dataset(inputs_test, 'test_inputs.npy')
save_dataset(targets_test, 'test_targets.npy')

tokenizer.save("tokenizer.pkl")


############################################################################################################################
############################################################################################################################


# Load and print the dataset to verify
print("SAMPLE FROM DATASET:")
inputs = np.load("train_inputs.npy", allow_pickle=True)
targets = np.load("train_targets.npy", allow_pickle=True)

random_index = random.randint(0, len(inputs) - 1)
line()
inputs = inputs.reshape(inputs.shape[0], -1)
targets = targets.reshape(targets.shape[0], -1)
tokenizer = CharTokenizer()
tokenizer.load('tokenizer.pkl')
print(f"{random_index}-th sentence (Input):")
print(f'"{tokenizer.decode(inputs[random_index].tolist())}"')
line()
print(f"{random_index}-th sentence (Target):")
print(f'"{tokenizer.decode(targets[random_index].tolist())}"')
line()