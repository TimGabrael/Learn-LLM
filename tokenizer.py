import torch
import re
from torch.utils.data import Dataset, DataLoader

class Tokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
    def vocab_size(self):
        return len(self.str_to_int)

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

    def vocab_size(self):
        return self.tokenizer.vocab_size()

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    vocab = {}
    with open("shrek3.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        all_words = sorted(list(set(re.split(r'([,.?_!"()\']|--|\s)', raw_text))))
        all_words.extend(["<|endoftext|>", "<|unk|>"])
        vocab = {token:integer for integer, token in enumerate(all_words)}

    tokenizer = Tokenizer(vocab)
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

def create_tokenizer():
    vocab = {}
    with open("shrek3.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        all_words = sorted(list(set(re.split(r'([,.?_!"()\']|--|\s)', raw_text))))
        all_words.extend(["<|endoftext|>", "<|unk|>"])
        vocab = {token:integer for integer, token in enumerate(all_words)}
    tokenizer = Tokenizer(vocab)
    return tokenizer




