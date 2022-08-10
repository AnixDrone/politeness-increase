import torch
from typing import Iterable, List
import torchtext
from torchtext.vocab import build_vocab_from_iterator


def tokenize_sentence(sentence, token_transform):
    return token_transform(sentence)


def vocab_transform(tokens: List[str], vocab: torchtext.vocab.Vocab):
    return [vocab.vocab.__getitem__(token) for token in tokens]


def tensor_transform(token_ids, bos_idx, eos_idx):
    return torch.cat((torch.tensor([bos_idx]), torch.tensor(token_ids), torch.tensor([eos_idx])))


def sentence_processing(sentence, vocab, token_transform, bos_idx, eos_idx):
    tokens = tokenize_sentence(sentence, token_transform)
    token_ids = vocab_transform(tokens, vocab)
    tensor_ids = tensor_transform(token_ids, bos_idx, eos_idx)
    return tensor_ids

def yield_tokens(data_iter: Iterable,token_transform):
    for data_sample in data_iter:
        yield token_transform(data_sample[1])
        
def build_vocab(data_series,token_transform,special_symbols):
    data_iterator = yield_tokens(data_series.iteritems(),token_transform)
    vocab = build_vocab_from_iterator(data_iterator,
                                                        min_freq=3,
                                                        specials=special_symbols,
                                                        special_first=True)
    vocab.set_default_index(special_symbols.index('<unk>'))
    return vocab