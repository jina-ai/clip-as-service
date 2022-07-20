# Originally from https://github.com/openai/CLIP. MIT License, Copyright (c) 2021 OpenAI

import io
import os
import hashlib
import shutil
import urllib
from typing import List

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


_S3_BUCKET = 'https://clip-as-service.s3.us-east-2.amazonaws.com/models/torch/'
_MODELS = {
    'RN50': ('RN50.pt', '9140964eaaf9f68c95aa8df6ca13777c'),
    'RN101': ('RN101.pt', 'fa9d5f64ebf152bc56a18db245071014'),
    'RN50x4': ('RN50x4.pt', '03830990bc768e82f7fb684cde7e5654'),
    'RN50x16': ('RN50x16.pt', '83d63878a818c65d0fb417e5fab1e8fe'),
    'RN50x64': ('RN50x64.pt', 'a6631a0de003c4075d286140fc6dd637'),
    'ViT-B/32': ('ViT-B-32.pt', '3ba34e387b24dfe590eeb1ae6a8a122b'),
    'ViT-B/16': ('ViT-B-16.pt', '44c3d804ecac03d9545ac1a3adbca3a6'),
    'ViT-L/14': ('ViT-L-14.pt', '096db1af569b284eb76b3881534822d9'),
    'ViT-L/14@336px': ('ViT-L-14-336px.pt', 'b311058cae50cb10fbfa2a44231c9473'),
}

MODEL_SIZE = {
    'RN50': 224,
    'RN101': 224,
    'RN50x4': 288,
    'RN50x16': 384,
    'RN50x64': 448,
    'ViT-B/32': 224,
    'ViT-B/16': 224,
    'ViT-L/14': 224,
    'ViT-L/14@336px': 336,
}


def md5file(filename: str):
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def _download(
    url: str,
    target_folder: str,
    md5sum: str = None,
    with_resume: bool = True,
    max_attempts: int = 3,
) -> str:
    os.makedirs(target_folder, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(target_folder, filename)

    if os.path.exists(download_target):
        if not os.path.isfile(download_target):
            raise FileExistsError(f'{download_target} exists and is not a regular file')

        actual_md5sum = md5file(download_target)
        if (not md5sum) or actual_md5sum == md5sum:
            return download_target

    from rich.progress import (
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    progress = Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
    )

    with progress:
        task = progress.add_task('download', filename=url, start=False)

        for _ in range(max_attempts):
            tmp_file_path = download_target + '.part'
            resume_byte_pos = (
                os.path.getsize(tmp_file_path) if os.path.exists(tmp_file_path) else 0
            )

            try:
                # resolve the 403 error by passing a valid user-agent
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                total_bytes = int(
                    urllib.request.urlopen(req).info().get('Content-Length', -1)
                )
                mode = 'ab' if (with_resume and resume_byte_pos) else 'wb'

                with open(tmp_file_path, mode) as output:
                    progress.update(task, total=total_bytes)
                    progress.start_task(task)

                    if resume_byte_pos and with_resume:
                        progress.update(task, advance=resume_byte_pos)
                        req.headers['Range'] = f'bytes={resume_byte_pos}-'

                    with urllib.request.urlopen(req) as source:
                        while True:
                            buffer = source.read(8192)
                            if not buffer:
                                break

                            output.write(buffer)
                            progress.update(task, advance=len(buffer))

                actual_md5 = md5file(tmp_file_path)
                if (md5sum and actual_md5 == md5sum) or (not md5sum):
                    shutil.move(tmp_file_path, download_target)
                    return download_target
                else:
                    os.remove(tmp_file_path)
                    raise RuntimeError(
                        f'MD5 mismatch: expected {md5sum}, got {actual_md5}'
                    )

            except Exception as ex:
                progress.console.print(
                    f'Failed to download {url} with {ex!r} at the {_}th attempt'
                )
                progress.reset(task)

        raise RuntimeError(
            f'Failed to download {url} within retry limit {max_attempts}'
        )


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


def available_models() -> List[str]:
    '''Returns the names of available CLIP models'''
    return list(_MODELS.keys())
