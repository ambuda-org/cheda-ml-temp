import torch
import sys

import numpy as np
from tqdm import tqdm

from tokenizer import CharTokenizer
from model import Transformer

import torch
import torch.nn as nn
import torch.optim as optim

#############################################################
#############################################################

tokenizer = CharTokenizer()
tokenizer.load('tokenizer.pkl')

UNK_TOKEN_ID = 0
PAD_TOKEN_ID = 1
SOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3

inputs_train = np.load("train_inputs.npy", allow_pickle=True)
targets_train = np.load("train_targets.npy", allow_pickle=True)
inputs_test = np.load("test_inputs.npy", allow_pickle=True)
targets_test = np.load("test_targets.npy", allow_pickle=True)

print("LOADED TOKENIZER AND DATASET......")

#############################################################
#############################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_model = True
load_model = False

# Training hyperparameters
num_epochs = 10000
learning_rate = 3e-4
batch_size = 64

# Model hyperparameters
src_vocab_size = tokenizer.next_id
trg_vocab_size = tokenizer.next_id
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 300
forward_expansion = 4
src_pad_idx = PAD_TOKEN_ID

step = 0

#############################################################
#############################################################

def get_batch(split):
    if split == 'train':
        inputs = inputs_train
        targets = targets_train
    else:
        inputs = inputs_test
        targets = targets_test

    ix = np.random.choice(len(inputs), batch_size)
    x = [inputs[i] for i in ix]
    y = [targets[i] for i in ix]

    x = np.array(x, dtype=np.int64)
    y = np.array(y, dtype=np.int64)

    x = torch.from_numpy(x).permute(1, 0)
    y = torch.from_numpy(y).permute(1, 0)
    
    x, y = x.to(device), y.to(device)

    return x, y

#############################################################
#############################################################

def load_checkpoint(checkpoint_path, model, optimizer):

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print("=> loaded checkpoint '{}'".format(checkpoint_path))

def save_checkpoint(checkpoint, filename='ckpt.pth'):
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

#############################################################
#############################################################

def transform_sentence(model, sentence, device, max_length=50):

    tokens = tokenizer.encode(sentence)
    sentence_tensor = torch.LongTensor(tokens)
    sentence_tensor = sentence_tensor.reshape(sentence_tensor.shape[0], 1)
    sentence_tensor = sentence_tensor.to(device)

    # Add <SOS> and <EOS> in beginning and end respectively
    outputs = [SOS_TOKEN_ID]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == EOS_TOKEN_ID:
            break

    translated_sentence = tokenizer.decode(outputs)
    
    return translated_sentence


#############################################################
#############################################################


model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = PAD_TOKEN_ID
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint("ckpt.pth", model, optimizer)

sentence = "hayozwraKaravaktrAS_ca_rAkzasIr_GoradarSanAHSUlamudgarahastAS_ca_kroDanAH_kalahapriyAH"
# hayozwraKaravaktrAS_ca_rAkzasIr_GoradarSanAHSUlamudgarahastAS_ca_kroDanAH_kalahapriyAH
# haya_uzwra_Kara_vaktrAH_ca_rAkzasIH_Gora_darSanAH_SUla_mudgara_hastAH_ca_kroDanAH_kalaha_priyAH


#############################################################
#############################################################

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = transform_sentence(
        model, sentence, device, max_length=300
    )

    print(f"Transformed example sentence: \n {translated_sentence}")
    print(f"Expected: \n {'haya_uzwra_Kara_vaktrAH_ca_rAkzasIH_Gora_darSanAH_SUla_mudgara_hastAH_ca_kroDanAH_kalaha_priyAH'}")
    
    model.train()
    
    losses = []

    num_batches = inputs_train.shape[0] // batch_size

    for _t in tqdm(range(num_batches), desc="Processing Batches"):
        x, y = get_batch('train')
        inp_data = x
        target = y

        # Forward prop
        output = model(inp_data, target[:-1, :])

        # (Some Magic)

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshaping.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        # Back prop
        loss.backward()

        # Healthy Gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        step += 1

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)