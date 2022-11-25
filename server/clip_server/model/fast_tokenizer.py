import os
import gzip
from functools import lru_cache
from rust_tokenizers import PyOpenAiGptTokenizer
from clip_server.helper import __resources_path__


@lru_cache()
def default_vocab():
    vocab_path = os.path.join(__resources_path__, 'vocab.json')
    if not os.path.isfile(vocab_path):
        import json
        from simple_tokenizer import bytes_to_unicode

        merges = open(default_merges()).read()
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<[startoftext>', '<endoftext>'])
        vocab = dict(zip(vocab, range(len(vocab))))
        print(vocab)
        with open(vocab_path, 'w') as f:
            f.write(json.dumps(vocab))
    return vocab_path


@lru_cache()
def default_merges():
    merges_path = os.path.join(__resources_path__, 'merges.txt')
    if not os.path.isfile(merges_path):
        bpe_path = os.path.join(__resources_path__, 'bpe_simple_vocab_16e6.txt.gz')
        merges_file = gzip.GzipFile(bpe_path)
        with open(merges_path, 'w') as f:
            f.write(merges_file.read().decode('utf-8'))
    return merges_path


class FastTokenizer(object):
    def __init__(
        self, vocab_path: str = default_vocab(), merges_path: str = default_merges()
    ):
        # TODO: <unk>
        self.tokenizer = PyOpenAiGptTokenizer(vocab_path, merges_path, True)

    def encode(self, text):
        return self.tokenizer.encode(text).token_ids
