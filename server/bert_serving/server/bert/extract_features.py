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

    def __init__(self, tokens, input_ids, input_mask, input_type_ids):
        # self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_lst_to_features(lst_str, max_seq_length, max_position_embeddings,
                            tokenizer, logger, is_tokenized=False, mask_cls_sep=False):
    """Loads a data file into a list of `InputBatch`s."""

    examples = read_tokenized_examples(lst_str) if is_tokenized else read_examples(lst_str)

    _tokenize = lambda x: tokenizer.mark_unk_tokens(x) if is_tokenized else tokenizer.tokenize(x)

    all_tokens = [(_tokenize(ex.text_a), _tokenize(ex.text_b) if ex.text_b else []) for ex in examples]

    # user did not specify a meaningful sequence length
    # override the sequence length by the maximum seq length of the current batch
    if max_seq_length is None:
        max_seq_length = max(len(ta) + len(tb) for ta, tb in all_tokens)
        # add special tokens into account
        # case 1: Account for [CLS], tokens_a [SEP], tokens_b [SEP] -> 3 additional tokens
        # case 2: Account for [CLS], tokens_a [SEP] -> 2 additional tokens
        max_seq_length += 3 if any(len(tb) for _, tb in all_tokens) else 2
        max_seq_length = min(max_seq_length, max_position_embeddings)
        logger.warning('"max_seq_length" is undefined, '
                       'and bert config json defines "max_position_embeddings"=%d. '
                       'hence set "max_seq_length"=%d according to the current batch.' % (
                           max_position_embeddings, max_seq_length))

    for (tokens_a, tokens_b) in all_tokens:
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

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
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        input_type_ids = [0] * len(tokens)
        input_mask = [int(not mask_cls_sep)] + [1] * len(tokens_a) + [int(not mask_cls_sep)]

        if tokens_b:
            tokens += tokens_b + ['[SEP]']
            input_type_ids += [1] * (len(tokens_b) + 1)
            input_mask += [1] * len(tokens_b) + [int(not mask_cls_sep)]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Zero-pad up to the sequence length. more pythonic
        pad_len = max_seq_length - len(input_ids)
        input_ids += [0] * pad_len
        input_mask += [0] * pad_len
        input_type_ids += [0] * pad_len

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(input_type_ids) == max_seq_length

        logger.debug('tokens: %s' % ' '.join([tokenization.printable_text(x) for x in tokens]))
        logger.debug('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
        logger.debug('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
        logger.debug('input_type_ids: %s' % ' '.join([str(x) for x in input_type_ids]))

        yield InputFeatures(
            # unique_id=example.unique_id,
            tokens=tokens,
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
