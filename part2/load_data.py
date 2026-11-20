import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch
from functools import lru_cache
PAD_IDX = 0

MODEL_NAME = "google-t5/t5-small"
INPUT_PREFIX = "translate English to SQL: "

# Standard pretrained T5 starts decoding with its pad token.
DECODER_START_IDX = PAD_IDX


@lru_cache(maxsize=1)
def get_tokenizer():
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)

    # Preserve multi-character SQL operators as whole tokens.
    # Ordinary tokens remain visible with skip_special_tokens=True.
    tokenizer.add_tokens(
        ["<=", ">=", "<>", "!=", "<"],
        special_tokens=False,
    )

    assert tokenizer.pad_token_id == PAD_IDX
    return tokenizer


def preprocess_nl(text):
    return INPUT_PREFIX + text.strip()


def preprocess_sql(text):
    return text.strip()
class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Unknown split: {split}")

        self.split = split
        self.tokenizer = get_tokenizer()
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        self.raw_inputs = load_lines(
            os.path.join(data_folder, f"{split}.nl")
        )

        if not self.raw_inputs or any(not x for x in self.raw_inputs):
            raise ValueError(f"{split}.nl is empty or contains blank examples.")

        self.inputs = [
            preprocess_nl(text) for text in self.raw_inputs
        ]

        self.encoder_ids = tokenizer(
            self.inputs,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]

        self.raw_targets = None
        self.targets = None
        self.target_ids = None

        # Test examples have no gold SQL targets.
        if split != "test":
            self.raw_targets = load_lines(
                os.path.join(data_folder, f"{split}.sql")
            )

            if len(self.raw_inputs) != len(self.raw_targets):
                raise ValueError(
                    f"{split}: question and SQL counts do not match."
                )

            if any(not x for x in self.raw_targets):
                raise ValueError(f"{split}.sql contains blank targets.")

            self.targets = [
                preprocess_sql(text) for text in self.raw_targets
            ]

            self.target_ids = tokenizer(
                self.targets,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
    
    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        encoder_ids = torch.tensor(
            self.encoder_ids[idx],
            dtype=torch.long,
        )

        if self.split == "test":
            return encoder_ids

        decoder_targets = torch.tensor(
            self.target_ids[idx],
            dtype=torch.long,
        )

        return encoder_ids, decoder_targets

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_sequences, target_sequences = zip(*batch)

    encoder_ids = pad_sequence(
        encoder_sequences,
        batch_first=True,
        padding_value=PAD_IDX,
    )
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # Shift targets right by one position:
    # targets:         token_1 token_2 ... EOS
    # decoder inputs:  START   token_1 ... token_last
    decoder_sequences = [
        torch.cat([
            target.new_tensor([DECODER_START_IDX]),
            target[:-1],
        ])
        for target in target_sequences
    ]

    decoder_inputs = pad_sequence(
        decoder_sequences,
        batch_first=True,
        padding_value=PAD_IDX,
    )

    # Match the supplied training loop, which masks PAD_IDX targets.
    decoder_targets = pad_sequence(
        target_sequences,
        batch_first=True,
        padding_value=PAD_IDX,
    )

    initial_decoder_inputs = torch.full(
        (len(batch), 1),
        DECODER_START_IDX,
        dtype=torch.long,
    )

    return (
        encoder_ids,
        encoder_mask,
        decoder_inputs,
        decoder_targets,
        initial_decoder_inputs,
    )
def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence(
        batch,
        batch_first=True,
        padding_value=PAD_IDX,
    )
    encoder_mask = (encoder_ids != PAD_IDX).long()

    initial_decoder_inputs = torch.full(
        (len(batch), 1),
        DECODER_START_IDX,
        dtype=torch.long,
    )

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))

    if len(train_x) != len(train_y):
        raise ValueError("Train question and SQL counts do not match.")

    if len(dev_x) != len(dev_y):
        raise ValueError("Dev question and SQL counts do not match.")
    return train_x, train_y, dev_x, dev_y, test_x