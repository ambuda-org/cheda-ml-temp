import pickle

# Define special token IDs
PAD_TOKEN_ID = 1
SOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3

class CharTokenizer:
    def __init__(self):
        self.char_to_id = {'<PAD>': PAD_TOKEN_ID, '<SOS>': SOS_TOKEN_ID, '<EOS>': EOS_TOKEN_ID}
        self.id_to_char = {PAD_TOKEN_ID: '<PAD>', SOS_TOKEN_ID: '<SOS>', EOS_TOKEN_ID: '<EOS>'}
        self.next_id = 4

    def encode(self, text):
        encoded = [SOS_TOKEN_ID]
        for char in text:
            if char not in self.char_to_id:
                self.char_to_id[char] = self.next_id
                self.id_to_char[self.next_id] = char
                self.next_id += 1
            encoded.append(self.char_to_id[char])
        encoded.append(EOS_TOKEN_ID)
        return encoded

    def decode(self, ids):
        return ''.join(self.id_to_char[id_] for id_ in ids if id_ not in (SOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID))

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump((self.char_to_id, self.id_to_char, self.next_id), file)

    def load(self, file_path):
        with open(file_path, 'rb') as file:
            self.char_to_id, self.id_to_char, self.next_id = pickle.load(file)
        return