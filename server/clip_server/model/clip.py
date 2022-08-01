# Originally from https://github.com/openai/CLIP. MIT License, Copyright (c) 2021 OpenAI

import io

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert('RGB')


def _blob2image(blob):
    return Image.open(io.BytesIO(blob))


def _transform_blob(n_px):
    return Compose(
        [
            _blob2image,
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def _transform_ndarray(n_px):
    return Compose(
        [
            ToTensor(),
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
