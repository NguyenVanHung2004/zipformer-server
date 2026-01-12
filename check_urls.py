import urllib.request
import urllib.error

urls = [
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-2023-09-11.tar.bz2",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-2024-03-07.tar.bz2",
    "https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2023-09-11/resolve/main/sherpa-onnx-zipformer-vi-2023-09-11.tar.bz2",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-bpe-zipformer-vi-2023-09-11.tar.bz2",
]

for url in urls:
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req) as response:
            print(f"✅ FOUND: {url} (Status: {response.status})")
    except urllib.error.HTTPError as e:
        print(f"❌ {url} - {e.code}")
    except Exception as e:
        print(f"❌ {url} - {e}")
