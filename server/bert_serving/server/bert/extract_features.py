# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re

from . import tokenization

__all__ = ['convert_lst_to_features']


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_type_ids):
        # self.unique_id = unique_id
        # self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_lst_to_features(lst_str, seq_length, tokenizer, is_tokenized=False):
    """Loads a data file into a list of `InputBatch`s."""

    examples = read_tokenized_examples(lst_str) if is_tokenized else read_examples(lst_str)

    _tokenize = lambda x: tokenizer.mark_unk_tokens(x) if is_tokenized else tokenizer.tokenize(x)

    for (ex_index, example) in enumerate(examples):
        tokens_a = _tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = _tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length. more pythonic
        pad_len = seq_length - len(input_ids)
        input_ids += [0] * pad_len
        input_mask += [0] * pad_len
        input_type_ids += [0] * pad_len

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        # print("*** Example ***")
        # print("unique_id: %s" % (example.unique_id))
        # print("tokens: %s" % " ".join(
        #     [tokenization.printable_text(x) for x in tokens]))
        # print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        # print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        # print("input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        yield InputFeatures(
            # unique_id=example.unique_id,
            # tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(lst_strs):
    """Read a list of `InputExample`s from a list of strings."""
    unique_id = 0
    for ss in lst_strs:
        line = tokenization.convert_to_unicode(ss)
        if not line:
            continue
        line = line.strip()
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        yield InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
        unique_id += 1


def read_tokenized_examples(lst_strs):
    unique_id = 0
    lst_strs = [[tokenization.convert_to_unicode(w) for w in s] for s in lst_strs]
    for ss in lst_strs:
        text_a = ss
        text_b = None
        try:
            j = ss.index('|||')
            text_a = ss[:j]
            text_b = ss[(j + 1):]
        except ValueError:
            pass
        yield InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
        unique_id += 1
