import os
import hashlib
import shutil
import urllib


_OPENCLIP_S3_BUCKET = 'https://clip-as-service.s3.us-east-2.amazonaws.com/models/torch'
_OPENCLIP_MODELS = {
    'RN50::openai': ('RN50.pt', '9140964eaaf9f68c95aa8df6ca13777c'),
    'RN50::yfcc15m': ('RN50-yfcc15m.pt', 'e9c564f91ae7dc754d9043fdcd2a9f22'),
    'RN50::cc12m': ('RN50-cc12m.pt', '37cb01eb52bb6efe7666b1ff2d7311b5'),
    'RN101::openai': ('RN101.pt', 'fa9d5f64ebf152bc56a18db245071014'),
    'RN101::yfcc15m': ('RN101-yfcc15m.pt', '48f7448879ce25e355804f6bb7928cb8'),
    'RN50x4::openai': ('RN50x4.pt', '03830990bc768e82f7fb684cde7e5654'),
    'RN50x16::openai': ('RN50x16.pt', '83d63878a818c65d0fb417e5fab1e8fe'),
    'RN50x64::openai': ('RN50x64.pt', 'a6631a0de003c4075d286140fc6dd637'),
    'ViT-B-32::openai': ('ViT-B-32.pt', '3ba34e387b24dfe590eeb1ae6a8a122b'),
    'ViT-B-32::laion2b_e16': (
        'ViT-B-32-laion2b_e16.pt',
        'df08de3d9f2dc53c71ea26e184633902',
    ),
    'ViT-B-32::laion400m_e31': (
        'ViT-B-32-laion400m_e31.pt',
        'ca8015f98ab0f8780510710681d7b73e',
    ),
    'ViT-B-32::laion400m_e32': (
        'ViT-B-32-laion400m_e32.pt',
        '359e0dba4a419f175599ee0c63a110d8',
    ),
    'ViT-B-32::laion2B-s34B-b79K': (
        'ViT-B-32-laion2B-s34B-b79K.bin',
        '2fc036aea9cd7306f5ce7ce6abb8d0bf',
    ),
    'ViT-B-16::openai': ('ViT-B-16.pt', '44c3d804ecac03d9545ac1a3adbca3a6'),
    'ViT-B-16::laion400m_e31': (
        'ViT-B-16-laion400m_e31.pt',
        '31306a44224cc46fec1bc3b82fd0c4e6',
    ),
    'ViT-B-16::laion400m_e32': (
        'ViT-B-16-laion400m_e32.pt',
        '07283adc5c17899f2ed22d82b563c54b',
    ),
    'ViT-B-16-plus-240::laion400m_e31': (
        'ViT-B-16-plus-240-laion400m_e31.pt',
        'c88f453644a998ecb094d878a2f0738d',
    ),
    'ViT-B-16-plus-240::laion400m_e32': (
        'ViT-B-16-plus-240-laion400m_e32.pt',
        'e573af3cef888441241e35022f30cc95',
    ),
    'ViT-L-14::openai': ('ViT-L-14.pt', '096db1af569b284eb76b3881534822d9'),
    'ViT-L-14::laion400m_e31': (
        'ViT-L-14-laion400m_e31.pt',
        '09d223a6d41d2c5c201a9da618d833aa',
    ),
    'ViT-L-14::laion400m_e32': (
        'ViT-L-14-laion400m_e32.pt',
        'a76cde1bc744ca38c6036b920c847a89',
    ),
    'ViT-L-14::laion2B-s32B-b82K': (
        'ViT-L-14-laion2B-s32B-b82K.bin',
        '4d2275fc7b2d7ee9db174f9b57ddecbd',
    ),
    'ViT-L-14-336::openai': ('ViT-L-14-336px.pt', 'b311058cae50cb10fbfa2a44231c9473'),
    'ViT-H-14::laion2B-s32B-b79K': (
        'ViT-H-14-laion2B-s32B-b79K.bin',
        '2aa6c46521b165a0daeb8cdc6668c7d3',
    ),
    'ViT-g-14::laion2B-s12B-b42K': (
        'ViT-g-14-laion2B-s12B-b42K.bin',
        '3bf99353f6f1829faac0bb155be4382a',
    ),
    # older version name format
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

_MULTILINGUALCLIP_MODELS = {
    'M-CLIP/XLM-Roberta-Large-Vit-B-32': (),
    'M-CLIP/XLM-Roberta-Large-Vit-L-14': (),
    'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus': (),
    'M-CLIP/LABSE-Vit-L-14': (),
}

_VISUAL_MODEL_IMAGE_SIZE = {
    'RN50': 224,
    'RN101': 224,
    'RN50x4': 288,
    'RN50x16': 384,
    'RN50x64': 448,
    'ViT-B-32': 224,
    'ViT-B-16': 224,
    'Vit-B-16Plus': 240,
    'ViT-B-16-plus-240': 240,
    'ViT-L-14': 224,
    'ViT-L-14-336': 336,
    'ViT-H-14': 224,
    'ViT-g-14': 224,
}


def md5file(filename: str):
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def get_model_url_md5(name: str):
    model_pretrained = _OPENCLIP_MODELS[name]
    if len(model_pretrained) == 0:  # not on s3
        return None, None
    else:
        return (_OPENCLIP_S3_BUCKET + '/' + model_pretrained[0], model_pretrained[1])


def download_model(
    url: str,
    target_folder: str = os.path.expanduser("~/.cache/clip"),
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
        " \n",  # divide this bar from Flow's bar
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
        task = progress.add_task('download', filename=filename, start=False)

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
