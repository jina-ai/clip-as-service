import os
from functools import lru_cache
from rust_tokenizers import PyOpenAiGptTokenizer
from clip_server.helper import __resources_path__


@lru_cache()
def default_vocab():
    return os.path.join(__resources_path__, 'vocab.json')


@lru_cache()
def default_merges():
    return os.path.join(__resources_path__, 'merges.txt')


class FastTokenizer(object):
    def __init__(
        self, vocab_path: str = default_vocab, merges_path: str = default_merges
    ):
        self.tokenizer = PyOpenAiGptTokenizer(vocab_path, merges_path, True)

    def encode(self, text):
        return self.tokenizer.encode(text).token_ids
