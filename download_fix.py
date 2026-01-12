import urllib.request
import tarfile
import os
import shutil
import time

def download(url, filename):
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename} ({os.path.getsize(filename)} bytes)")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        raise

if not os.path.exists("model_vi"):
    os.makedirs("model_vi")

# 1. VAD
download("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx", "model_vi/silero_vad.onnx")

# 2. ASR
asr_file = "asr_model.tar.bz2"
download("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-2025-04-20.tar.bz2", asr_file)

print("Extracting ASR...")
with tarfile.open(asr_file, "r:bz2") as tar:
    tar.extractall(".")

extracted_dir = "sherpa-onnx-zipformer-vi-2025-04-20"
if os.path.exists(extracted_dir):
    for f in os.listdir(extracted_dir):
        src = os.path.join(extracted_dir, f)
        dst = os.path.join("model_vi", f)
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, "model_vi")
    os.rmdir(extracted_dir)
    print("Moved ASR files to model_vi")

if os.path.exists(asr_file):
    os.remove(asr_file)
print("Done.")
