import pytest
from clip_server.model.tokenization import Tokenizer


@pytest.mark.parametrize(
    "name", ["ViT-L/14@336px", "M-CLIP/XLM-Roberta-Large-Vit-B-32"]
)
def test_tokenizer_name(name):
    tokenizer = Tokenizer(name)

    result = tokenizer("hello world")
    assert result["input_ids"].shape == result["attention_mask"].shape
    assert result["input_ids"].shape[0] == 1

    result = tokenizer(["hello world", "welcome to the world"])
    assert result["input_ids"].shape == result["attention_mask"].shape
    assert result["input_ids"].shape[0] == 2
