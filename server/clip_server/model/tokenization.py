import torch
from typing import List, Union
from clip_server.model.pretrained_models import _MULTILINGUALCLIP_MODELS


class Tokenizer:
    def __init__(self, name: str, **kwargs):
        self._name = name
        if name in _MULTILINGUALCLIP_MODELS:
            import transformers

            self._tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        else:
            from clip_server.model.simple_tokenizer import SimpleTokenizer

            self._tokenizer = SimpleTokenizer()

    def encode(
        self,
        texts: Union[str, List[str]],
        context_length: int = 77,
        truncate: bool = True,
    ) -> dict:
        """
        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize

        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length

        Returns
        -------
        A dict of tokenized representations of the input strings and their corresponding attention masks with both
            shape = [batch size, context_length]
        """
        if isinstance(texts, str):
            texts = [texts]
        if self._name in _MULTILINGUALCLIP_MODELS:
            result = self._tokenizer(
                texts,
                max_length=context_length,
                return_attention_mask=True,
                return_tensors='pt',
                padding='max_length',
            )
            return {
                'input_ids': result['input_ids'],
                'attention_mask': result['attention_mask'],
            }
        else:
            sot_token = self._tokenizer.encoder['<|startoftext|>']
            eot_token = self._tokenizer.encoder['<|endoftext|>']
            all_tokens = [
                [sot_token] + self._tokenizer.encode(text) + [eot_token]
                for text in texts
            ]

            input_ids = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
            attention_mask = torch.zeros(
                len(all_tokens), context_length, dtype=torch.long
            )

            for i, tokens in enumerate(all_tokens):
                if len(tokens) > context_length:
                    if truncate:
                        tokens = tokens[:context_length]
                        tokens[-1] = eot_token
                    else:
                        raise RuntimeError(
                            f'Input {texts[i]} is too long for context length {context_length}'
                        )
                input_ids[i, : len(tokens)] = torch.tensor(tokens)
                attention_mask[i, : len(tokens)] = 1

            return {'input_ids': input_ids, 'attention_mask': attention_mask}
