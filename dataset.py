import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Literal
import os


class Multi30kDe2En(Dataset):
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']

    def __init__(self, split: Literal['train', 'valid']):
        super().__init__()

        assert split in ['train', 'valid']
        split_map = {
            'train': 'train',
            'valid': 'val'
        }

        base_path = os.path.join(
            'data', 'multi30k', 'data', 'task1', 'raw'
        )

        de_path = os.path.join(base_path, f"{split_map[split]}.de")
        en_path = os.path.join(base_path, f"{split_map[split]}.en")

        with open(de_path, encoding='utf-8') as f:
            self.de_texts = f.read().splitlines()

        with open(en_path, encoding='utf-8') as f:
            self.en_texts = f.read().splitlines()

        assert len(self.de_texts) == len(self.en_texts)

        self.de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
        self.en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

        # vocab chỉ build từ train
        if split == 'train':
            self.de_vocab, self.en_vocab = self._build_vocab()
        else:
            vocab_data = torch.load('vocab.pt')
            self.de_vocab = vocab_data['de']
            self.en_vocab = vocab_data['en']

    def __len__(self):
        return len(self.de_texts)

    def __getitem__(self, idx):
        de_tokens = self.de_tokenizer(self.de_texts[idx])
        en_tokens = self.en_tokenizer(self.en_texts[idx])

        de_ids = torch.tensor(
            [self.de_vocab[token] for token in de_tokens],
            dtype=torch.long
        )

        en_ids = torch.tensor(
            [self.en_vocab[token] for token in en_tokens],
            dtype=torch.long
        )

        return de_ids, en_ids

    def _build_vocab(self):
        de_tokens = (self.de_tokenizer(s) for s in self.de_texts)
        en_tokens = (self.en_tokenizer(s) for s in self.en_texts)

        de_vocab = build_vocab_from_iterator(
            de_tokens,
            specials=self.SPECIAL_SYMBOLS
        )
        en_vocab = build_vocab_from_iterator(
            en_tokens,
            specials=self.SPECIAL_SYMBOLS
        )

        de_vocab.set_default_index(self.UNK_IDX)
        en_vocab.set_default_index(self.UNK_IDX)

        torch.save(
            {'de': de_vocab, 'en': en_vocab},
            'vocab.pt'
        )

        return de_vocab, en_vocab

    @classmethod
    def collate_fn(cls, batch):
        de_batch, en_batch = [], []

        for de, en in batch:
            de = torch.cat([
                torch.tensor([cls.BOS_IDX]),
                de,
                torch.tensor([cls.EOS_IDX])
            ])
            en = torch.cat([
                torch.tensor([cls.BOS_IDX]),
                en,
                torch.tensor([cls.EOS_IDX])
            ])

            de_batch.append(de)
            en_batch.append(en)

        de_batch = pad_sequence(
            de_batch,
            padding_value=cls.PAD_IDX,
            batch_first=True
        )

        en_batch = pad_sequence(
            en_batch,
            padding_value=cls.PAD_IDX,
            batch_first=True
        )

        return de_batch, en_batch
