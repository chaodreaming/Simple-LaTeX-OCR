import requests
import os
import tqdm
import io
from pathlib import Path

url = 'https://github.com/chaodreaming/Simple-LaTeX-OCR/releases/latest'


def get_latest_tag():
    r = requests.get(url)
    tag = r.url.split('/')[-1]
    if tag == 'releases':
        return 'v0.0.1'
    return tag


def download_as_bytes_with_progress(url: str, name: str = None) -> bytes:
    # source: https://stackoverflow.com/questions/71459213/requests-tqdm-to-a-variable
    resp = requests.get(url, stream=True, allow_redirects=True)
    total = int(resp.headers.get('content-length', 0))
    bio = io.BytesIO()
    if name is None:
        name = url
    with tqdm.tqdm(
        desc=name,
        total=total,
        unit='b',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            bar.update(len(chunk))
            bio.write(chunk)
    return bio.getvalue()


def download_checkpoints():
    tag = 'v0.0.1'  # get_latest_tag()
    path = Path(__file__).parent.parent/"models"
    if not os.path.exists(path):
        os.makedirs(path)
    config="https://github.com/chaodreaming/Simple-LaTeX-OCR/releases/download/%s/config.yaml" % tag
    best='https://github.com/chaodreaming/Simple-LaTeX-OCR/releases/download/%s/best.onnx' % tag
    encoder='https://github.com/chaodreaming/Simple-LaTeX-OCR/releases/download/%s/decoder.onnx' % tag
    decoder='https://github.com/chaodreaming/Simple-LaTeX-OCR/releases/download/%s/encoder.onnx' % tag
    tokenizer='https://github.com/chaodreaming/Simple-LaTeX-OCR/releases/download/%s/tokenizer.json' % tag
    for url, name in zip([config,tokenizer,best,encoder,decoder], ["config.yaml","tokenizer.json","best.onnx", "decoder.onnx","encoder.onnx"]):
        download_path=os.path.join(path, name)
        if not os.path.exists(download_path):
            print('download ', tag," ",name, 'to path', download_path)
            file = download_as_bytes_with_progress(url, name)
            open(download_path, "wb").write(file)


if __name__ == '__main__':
    download_checkpoints()
