import asyncio
import websockets
import sherpa_onnx
import os
import json
import numpy as np
import logging
import sys
from collections import deque
import glob
import urllib.request
import tarfile
import shutil

# --- AUTO-DOWNLOAD MODELS (For No-Dockerfile Deployment) ---
def check_and_download_models():
    # 1. Define URLs
    asr_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-2025-04-20.tar.bz2"
    vad_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
    
    # 2. Check & Download VAD
    if not os.path.exists("model_vi/silero_vad.onnx"):
        print("⏳ Downloading VAD model...")
        os.makedirs("model_vi", exist_ok=True)
        try:
            urllib.request.urlretrieve(vad_url, "model_vi/silero_vad.onnx")
            print("✅ VAD Downloaded")
        except Exception as e:
            print(f"❌ VAD Download Failed: {e}")

    # 3. Check & Download ASR
    # Check if ANY encoder file exists
    if not glob.glob("model_vi/encoder-*.onnx"):
        print(f"⏳ Downloading ASR Model from {asr_url}...")
        try:
            filename = "asr_model.tar.bz2"
            urllib.request.urlretrieve(asr_url, filename)
            print("📦 Extracting ASR...")
            with tarfile.open(filename, "r:bz2") as tar:
                tar.extractall(".")
            
            # Move files from 'sherpa-onnx-zipformer-vi-2025-04-20' to 'model_vi'
            extracted_dir = "sherpa-onnx-zipformer-vi-2025-04-20"
            if os.path.exists(extracted_dir):
                os.makedirs("model_vi", exist_ok=True) 
                
                # Delete existing .onnx files in model_vi to avoid conflicts
                for f in glob.glob("model_vi/*.onnx"):
                    os.remove(f)
                
                for f in os.listdir(extracted_dir):
                    shutil.move(os.path.join(extracted_dir, f), "model_vi")
                os.rmdir(extracted_dir)
            
            if os.path.exists(filename): os.remove(filename)
            print("✅ ASR Model Ready")
        except Exception as e:
            print(f"❌ ASR Download Failed: {e}")

# Run check immediately
check_and_download_models()



# --- CONFIGURATION ---
PORT = int(os.environ.get("PORT", 6006))
# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_components():
    model_dir = "./model_vi"
    
    # Dynamic file finding
    tokens_path = os.path.join(model_dir, "tokens.txt")
    enc_files = glob.glob(os.path.join(model_dir, "encoder-*.onnx"))
    dec_files = glob.glob(os.path.join(model_dir, "decoder-*.onnx"))
    join_files = glob.glob(os.path.join(model_dir, "joiner-*.onnx"))
    
    if not (enc_files and dec_files and join_files):
         logging.error(f"❌ Could not find model files in {model_dir}")
         sys.exit(1)
         
    encoder_path = enc_files[0]
    decoder_path = dec_files[0]
    joiner_path = join_files[0]
    
    vad_model = os.path.join(model_dir, "silero_vad.onnx")

    if not all(os.path.exists(p) for p in [tokens_path, encoder_path, decoder_path, joiner_path, vad_model]):
        logging.error(f"❌ Thiếu model files! Kiểm tra thư mục '{model_dir}'.")
        sys.exit(1)

    logging.info("⏳ Đang tải Offline Recognizer...")
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens_path,
        encoder=encoder_path,
        decoder=decoder_path,
        joiner=joiner_path,
        num_threads=4,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    )
    
    logging.info("⏳ Đang tải VAD...")
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = vad_model
    vad_config.sample_rate = 16000
    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)

    logging.info("✅ Hệ thống đã sẵn sàng!")
    return recognizer, vad

recognizer, vad = create_components()

async def handle_connection(websocket):
    logging.info("🔗 Client đã kết nối")
    
    # Mỗi client cần một instance VAD riêng biệt nếu muốn stateful chính xác, 
    # nhưng sherpa_onnx.VoiceActivityDetector có vẻ giữ state buffer. 
    # Tuy nhiên, doc mẫu chỉ dùng 1 global vad nếu đơn luồng. 
    # Đa luồng: Tốt nhất nên tạo VAD mới cho mỗi conn hoặc đảm bảo thread-safe.
    # Để an toàn và đơn giản, ta sẽ tạo lại VAD cho mỗi connection hoặc reset.
    # Nhưng VAD load model cũng nhẹ. Ta sẽ init lại config clone từ global hoặc làm mới.
    
    # RE-INIT VAD for each client to avoid buffer mixing
    model_dir = "./model_vi"
    vad_model = os.path.join(model_dir, "silero_vad.onnx")
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = vad_model
    vad_config.sample_rate = 16000
    
    # ⚡ TUNING VAD PARAMETERS (NOISE REDUCTION)
    # Tăng threshold lên 0.6 để lọc tiếng chuột/phím (chỉ giọng nói rõ mới bắt)
    vad_config.silero_vad.threshold = 0.6         
    vad_config.silero_vad.min_silence_duration = 0.5 
    # Tăng min_speech lên 0.5s để bỏ qua tiếng click ngắn
    vad_config.silero_vad.min_speech_duration = 0.5 
    
    client_vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)
    
    # 🎧 BUFFERING LOGIC (Để bắt lại đoạn đầu bị mất)
    # Lưu giữ 0.5 giây âm thanh trước đó (16000 * 0.5 = 8000 mẫu)
    from collections import deque
    # Mỗi chunk từ client là 1 lượng samples nhất định, ta lưu raw samples vào deque
    # Tuy nhiên deque lưu từng item, nếu item là chunk to thì khó quản lý size chính xác.
    # Ta sẽ lưu list các array, và estimte size.
    # Đơn giản hơn: lưu 1 buffer vòng tròn bằng numpy array nhưng tốn chi phí copy.
    # Cách hiệu quả: Deque chứa các chunk, tổng duration ~0.5s.
    
    pre_speech_buffer = deque(maxlen=20) # Giả sử mỗi chunk ~50ms -> 20 chunks = 1s
    
    try:
        async for message in websocket:
            # message là bytes (audio chunk)
            samples = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Thêm vào buffer lịch sử
            pre_speech_buffer.append(samples)
            
            client_vad.accept_waveform(samples)
            
            while not client_vad.empty():
                speech_segment = client_vad.front.samples
                # [NEW] Lấy offset của segment này trong cả chuỗi streaming
                # client_vad.front.start là index sample bắt đầu của segment
                segment_offset_seconds = 0.0
                if hasattr(client_vad.front, 'start'):
                     segment_offset_seconds = client_vad.front.start / 16000.0
                
                client_vad.pop()
                
                if len(speech_segment) < 1000: # Bỏ qua đoạn quá ngắn (< 0.06s)
                    continue
                
                # ... (Padding code omitted for brevity, ensure we adjust offset if padding is used? 
                # Actually padding is prepended *before* this segment in my logic, 
                # but physically concatenating it changes the relative time in `recognizer`.
                # If I prepend history, the recognizer sees [HISTORY + SEGMENT].
                # The Recognizer timestamps start at 0.
                # So relative to stream:
                # Real start = segment_offset_seconds - duration(history)
                # Let's handle this carefully.
                
                prepend_duration = 0.0
                if pre_speech_buffer:
                    history_samples = np.concatenate(list(pre_speech_buffer))
                    if len(history_samples) > 8000:
                        history_samples = history_samples[-8000:]
                    
                    prepend_duration = len(history_samples) / 16000.0
                    speech_segment = np.concatenate((history_samples, speech_segment))
                
                logging.info(f"🗣️ Phát hiện tiếng nói ({len(speech_segment)/16000:.2f}s). Offset: {segment_offset_seconds:.2f}s")
                
                stream = recognizer.create_stream()
                stream.accept_waveform(16000, speech_segment)
                recognizer.decode_stream(stream)
                result = stream.result
                
                # [FIX]: Reconstruct text from tokens
                if hasattr(result, 'tokens'):
                    raw_tokens = result.tokens
                    reconstructed_text = "".join(raw_tokens).replace('▁', ' ').strip()
                    import re
                    text = re.sub(r'\s+', ' ', reconstructed_text)
                else:
                    text = result.text.strip()

                if text:
                    words = []
                    if hasattr(result, 'tokens') and hasattr(result, 'timestamps'):
                        for i, token in enumerate(result.tokens):
                            # Timestamp từ recognizer là relative so với đầu speech_segment (đã gồm padding)
                            local_start = result.timestamps[i]
                            
                            # Chuyển sang Absolute Timestamp
                            # Absolute = Segment_Start_In_Stream - Prepend_Duration + Local_Start
                            absolute_start = segment_offset_seconds - prepend_duration + local_start
                            
                            # Đảm bảo không âm
                            absolute_start = max(0.0, absolute_start)
                            
                            start = absolute_start
                            end = start + 0.1
                            if i < len(result.timestamps) - 1:
                                next_local = result.timestamps[i+1]
                                next_absolute = segment_offset_seconds - prepend_duration + next_local
                                end = next_absolute
                            
                            clean_word = token.replace('▁', '').strip()
                            words.append({
                                "word": clean_word,
                                "start": start,
                                "end": end,
                                "confidence": 1.0,
                                "speaker": 0
                            })

                    # Deepgram-compatible format
                    response = {
                        "channel": {
                            "alternatives": [
                                {
                                    "transcript": text,
                                    "confidence": 1.0,
                                    "words": words
                                }
                            ]
                        },
                        "is_final": True
                    }
                    logging.info(f"📝 Kết quả: {text}")
                    await websocket.send(json.dumps(response, ensure_ascii=False))

    except websockets.exceptions.ConnectionClosed:
        logging.info("🔌 Client đã ngắt kết nối")
    except Exception as e:
        logging.error(f"❌ Lỗi connection: {e}")

async def main():
    server = await websockets.serve(handle_connection, "0.0.0.0", PORT)
    logging.info(f"🚀 Server (VAD + Offline) đang lắng nghe tại ws://0.0.0.0:{PORT}")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())