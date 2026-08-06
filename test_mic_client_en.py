"""
Offline test for the Vietnamese fine-tuned Zipformer model
(the same model that `server.py` loads at runtime).

NOTE: Despite the `_en` suffix in the filename, this file actually
exercises the VI fine-tuned model — the previous paths pointed at
`model_ft/` which is no longer used. English recognition is tested
via `test_mic_client.py` with `?language=en` query.

Usage:
    python test_mic_client_en.py
"""
import sherpa_onnx
import wave
import numpy as np

# Same paths as server.py (TOKENS_VI / ENCODER_VI / DECODER_VI / JOINER_VI)
TOKENS_PATH  = "model_vi/tokens.txt"
ENCODER_PATH = "model_vi_fine_tune/local/encoder-epoch-3-avg-1_scaled_0.1.onnx"
DECODER_PATH = "model_vi_fine_tune/local/decoder-epoch-3-avg-1_scaled_0.1.onnx"
JOINER_PATH  = "model_vi_fine_tune/local/joiner-epoch-3-avg-1_scaled_0.1.onnx"

WAV_PATH = "wav_1.wav"  # đổi sang file .wav bất kỳ (16 kHz mono PCM 16-bit)


def main():
    print("🚀 1. Khởi tạo mô hình ONNX...")
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=TOKENS_PATH,
        encoder=ENCODER_PATH,
        decoder=DECODER_PATH,
        joiner=JOINER_PATH,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    )
    print("✅ Load model ONNX thành công!")

    print(f"\n🎧 2. Đọc file audio: {WAV_PATH}")
    with wave.open(WAV_PATH, "rb") as f:
        sample_rate = f.getframerate()
        frames = f.readframes(f.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        print(f"   Độ dài: {len(samples)/sample_rate:.2f} giây")

    print("\n🧠 3. Đưa vào ONNX dự đoán (Không qua bộ lọc nào)...")
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    result = stream.result

    print("\n" + "=" * 50)
    print("🎯 KẾT QUẢ CUỐI CÙNG (FINAL TEXT):")
    print(f"[{result.text}]")
    print("=" * 50)

    print("\n🔍 4. MỔ XẺ TỪNG TOKEN BÊN TRONG ONNX:")
    if hasattr(result, "tokens") and hasattr(result, "timestamps"):
        for token, timestamp in zip(result.tokens, result.timestamps):
            print(f" - Token: '{token:<10}' | Giây thứ: {timestamp:.2f}s")
    else:
        print("Mô hình không trả về token/timestamp.")


if __name__ == "__main__":
    main()
