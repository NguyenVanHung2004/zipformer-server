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

# --- AUTO-DOWNLOAD MODELS (Robust Version) ---
def download_file(url, target_path, min_size=1024):
    print(f"⏳ Downloading {url} to {target_path}...")
    try:
        # Use simple urlretrieve but verify size afterwards
        urllib.request.urlretrieve(url, target_path)
        
        size = os.path.getsize(target_path)
        if size < min_size:
            print(f"❌ File too small ({size} bytes). Probable 404/Error page. Deleting...")
            os.remove(target_path)
            return False
            
        print(f"✅ Downloaded ({size/1024:.2f} KB)")
        return True
    except Exception as e:
        print(f"❌ Download Failed: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)
        return False

def check_and_download_models():
    # 1. Define URLs
    asr_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-2025-04-20.tar.bz2"
    # Alternative Mirror for VAD if GitHub fails
    vad_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
    
    # 2. Check & Download VAD
    if not os.path.exists("model_vi/silero_vad.onnx") or os.path.getsize("model_vi/silero_vad.onnx") < 1024:
        os.makedirs("model_vi", exist_ok=True)
        # Try primary URL
        if not download_file(vad_url, "model_vi/silero_vad.onnx"):
            # Try mirror if primary fails
            print("⚠️ Retrying with mirror...")
            download_file("https://huggingface.co/csukuangfj/silero-vad-onnx/resolve/main/silero_vad.onnx", "model_vi/silero_vad.onnx")

    # 3. Check & Download ASR
    if not glob.glob("model_vi/encoder-*.onnx"):
        filename = "asr_model.tar.bz2"
        if download_file(asr_url, filename):
            print("📦 Extracting ASR...")
            try:
                with tarfile.open(filename, "r:bz2") as tar:
                    tar.extractall(".")
                
                extracted_dir = "sherpa-onnx-zipformer-vi-2025-04-20"
                if os.path.exists(extracted_dir):
                    os.makedirs("model_vi", exist_ok=True) 
                    
                    # Delete existing .onnx files in model_vi (EXCEPT VAD)
                    for f in glob.glob("model_vi/*.onnx"):
                        if "silero_vad" not in f:
                            os.remove(f)
                    
                    for f in os.listdir(extracted_dir):
                        shutil.move(os.path.join(extracted_dir, f), "model_vi")
                    os.rmdir(extracted_dir)
                
                print("✅ ASR Model Ready")
            except Exception as e:
                print(f"❌ Extraction Failed: {e}")
            finally:
                if os.path.exists(filename): os.remove(filename)

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

    missing_files = []
    for p in [tokens_path, encoder_path, decoder_path, joiner_path, vad_model]:
        if not os.path.exists(p):
            missing_files.append(p)
            
    if missing_files:
        logging.error(f"❌ Thiếu model files: {missing_files}")
        logging.error(f"Contents of {model_dir}: {os.listdir(model_dir)}")
        sys.exit(1)

    logging.info("⏳ Đang tải Offline Recognizer...")
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens_path,
        encoder=encoder_path,
        decoder=decoder_path,
        joiner=joiner_path,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="modified_beam_search",
    )
    
    logging.info("⏳ Đang tải VAD...")
    
    # [DEBUG] Check file header
    if os.path.exists(vad_model):
        try:
            with open(vad_model, "rb") as f:
                header = f.read(64)
                logging.info(f"🔍 VAD Header (hex): {header.hex()}")
                logging.info(f"🔍 VAD Header (text): {header}")
        except Exception as e:
             logging.error(f"❌ Cannot read VAD header: {e}")

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
    import uuid
    conn_id = str(uuid.uuid4())[:8]
    logging.info(f"🔗 [{conn_id}] Client đã kết nối")
    
    model_dir = "./model_vi"
    vad_model = os.path.join(model_dir, "silero_vad.onnx")
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = vad_model
    vad_config.sample_rate = 16000
    
    # ⚡ TUNING VAD PARAMETERS (SENSITIVITY BOOST)
    # Threshold 0.35: Nhạy hơn để bắt giọng nói nhỏ -> Buffer sẽ chứa đủ âm thanh
    # Silence 1.0s: Chờ lâu hơn để chắc chắn hết câu -> Tránh cắt giữa chừng
    vad_config.silero_vad.threshold = 0.35
    vad_config.silero_vad.min_silence_duration = 1.0
    vad_config.silero_vad.min_speech_duration = 0.1 
    
    client_vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)
    
    # --- PSEUDO-STREAMING LOGIC ---
    rolling_buffer = [] 
    last_decode_time = 0
    DECODE_INTERVAL = 0.4 
    
    current_sentence_id = 0
    current_speaker = 0 
    last_segment_end_time = 0 
    
    # [TIMESTAMP FIX]
    total_samples_processed = 0 
    vad_start_offset_samples = 0

    is_partial_decoding = False
    
    async def run_partial_decode(buffer_copy, sentence_id):
        nonlocal is_partial_decoding, last_decode_time, current_sentence_id
        try:
            if sentence_id != current_sentence_id:
                return

            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, decode_buffer_sync, recognizer, buffer_copy)
            
            if sentence_id != current_sentence_id:
                return
            
            if text:
                 response = {
                    "channel": {
                        "alternatives": [{
                            "transcript": text,
                            "confidence": 0.5,
                        }]
                    },
                    "is_final": False
                }
                 try:
                    await websocket.send(json.dumps(response, ensure_ascii=False))
                 except:
                     pass 
            
            last_decode_time = asyncio.get_event_loop().time()
            
        except Exception as e:
            logging.error(f"⚠️ Partial Decode Error: {e}")
        finally:
            is_partial_decoding = False

    try:
        async for message in websocket:
            samples = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
            total_samples_processed += len(samples)
            
            max_amp = np.max(np.abs(samples)) if len(samples) > 0 else 0
            
            # [REMOVED NOISE GATE] Xử lý mọi âm thanh để không mất dữ liệu "đuôi" câu
            if max_amp > 0 and max_amp < 0.3: 
                target_gain = 0.5 / max_amp 
                perform_gain = min(target_gain, 4.0) 
                samples = samples * perform_gain
            
            samples = np.clip(samples, -1.0, 1.0)
            
            # 1. Luôn thêm vào buffer (để Partial & Final giống hệt nhau)
            rolling_buffer.extend(samples)
            
            # 2. Vẫn feed VAD để phát hiện sự im lặng (Trigger)
            client_vad.accept_waveform(samples)
            
            # [DEBUG]
            pass

            current_time = asyncio.get_event_loop().time()
            
            # A. CHECK FOR FINAL SEGMENTS (BUFFER-BASED STRATEGY)
            # VAD chỉ đóng vai trò là "Cò" (Trigger). Khi VAD báo hết câu (có segment):
            # -> Ta lấy TOÀN BỘ rolling_buffer ra decode. Đảm bảo Final == Partial.
            if not client_vad.empty():
                logging.info(f"⚡ VAD Triggered End of Sentence")
                
                # 1. Xả hết sự kiện VAD (nhưng không dùng audio trong đó)
                while not client_vad.empty():
                    client_vad.pop()
                
                # 2. Decode FULL Buffer
                if len(rolling_buffer) > 1600: 
                    stream = recognizer.create_stream()
                    stream.accept_waveform(16000, np.array(rolling_buffer, dtype=np.float32))
                    recognizer.decode_stream(stream)
                    result = stream.result
                    
                    text = result.text.strip().lower()
                    
                    # [TIMESTAMP FIX] Tính thời gian bắt đầu dựa trên tổng sample trôi qua
                    # buffer_start_time = Hiện tại - Độ dài buffer
                    buffer_start_time = (total_samples_processed - len(rolling_buffer)) / 16000.0
                    
                    if text:
                        # [SMART PARAGRAPHING]
                        time_gap = current_time - last_segment_end_time
                        if last_segment_end_time > 0 and time_gap > 2.0:
                            current_speaker = 1 - current_speaker 
                            logging.info(f"¶ New Paragraph (Gap: {time_gap:.2f}s)")
                            
                        last_segment_end_time = current_time 

                        # [FEATURE] Word-level Timestamps
                        words = []
                        if hasattr(result, 'tokens') and hasattr(result, 'timestamps'):
                            for i, token in enumerate(result.tokens):
                                 local_start = result.timestamps[i]
                                 absolute_start = buffer_start_time + local_start 
                                 
                                 start = round(absolute_start, 2)
                                 end = round(start + 0.1, 2)
                                 
                                 if i < len(result.timestamps) - 1:
                                     next_local = result.timestamps[i+1]
                                     next_absolute = buffer_start_time + next_local
                                     end = round(next_absolute, 2)
                                 
                                 clean_word = token.replace('▁', '').strip().lower()
                                 words.append({
                                     "word": clean_word,
                                     "start": start,
                                     "end": end,
                                     "confidence": 1.0,
                                     "speaker": current_speaker
                                 })
                        
                        if not words:
                            words.append({
                                "word": text,
                                "start": round(buffer_start_time, 2),
                                "end": round(buffer_start_time + (len(rolling_buffer)/16000), 2),
                                "confidence": 1.0,
                                "speaker": current_speaker
                            })
                        
                        logging.info(f"✅ [{conn_id}] Final Result [Speaker {current_speaker}]: {text}")
                        response = {
                            "channel": {
                                "alternatives": [{
                                    "transcript": text,
                                    "confidence": 1.0,
                                    "speaker": current_speaker, 
                                    "words": words 
                                }]
                            },
                            "is_final": True
                        }
                        await websocket.send(json.dumps(response, ensure_ascii=False))
                    
                # 3. Reset Buffer (Giữ lại đuôi nhỏ 0.25s để tránh cắt quá gắt nếu người dùng nói nối)
                if len(rolling_buffer) > 4000:
                    rolling_buffer = rolling_buffer[-4000:]
                else:
                    rolling_buffer = [] 
                
                current_sentence_id += 1  
                
            # [CRITICAL FEATURE] C. FORCED SEGMENTATION (Prevent Freezing on Long Speech)
            # If user speaks continuously for > 8 seconds without silence, FORCE a cut.
            # Set to 8s as a balanced limit
            if len(rolling_buffer) > 128000: # 16000 * 8s
                 logging.info(f"⚠️ Forced Segmentation (Long Speech > 8s)")
                 stream = recognizer.create_stream()
                 stream.accept_waveform(16000, np.array(rolling_buffer, dtype=np.float32))
                 recognizer.decode_stream(stream)
                 # [FIX] Lowercase forced segment
                 text = stream.result.text.strip().lower()
                 
                 # [TIMESTAMP FIX] Calculate Buffer Start Time
                 buffer_start_time = (total_samples_processed - len(rolling_buffer)) / 16000.0
                 
                 if text:
                    logging.info(f"✅ Final Result (Forced - Speaker {current_speaker}): {text}")
                    
                    # [FEATURE] Word-level Timestamps for Forced Segment
                    words_forced = []
                    result = stream.result
                    
                    if hasattr(result, 'tokens') and hasattr(result, 'timestamps'):
                        for i, token in enumerate(result.tokens):
                             local_start = result.timestamps[i]
                             absolute_start = buffer_start_time + local_start
                             
                             start = absolute_start
                             end = start + 0.1
                             
                             if i < len(result.timestamps) - 1:
                                 next_local = result.timestamps[i+1]
                                 next_absolute = buffer_start_time + next_local
                                 end = next_absolute
                             
                             clean_word = token.replace('▁', '').strip().lower()
                             words_forced.append({
                                 "word": clean_word,
                                 "start": round(start, 2),
                                 "end": round(end, 2),
                                 "confidence": 1.0,
                                 "speaker": current_speaker
                             })

                    # Fallback if no tokens
                    if not words_forced:
                        words_forced = [{
                            "word": text,
                            "start": round(buffer_start_time, 2),
                            "end": round(buffer_start_time + 8.0, 2),
                            "confidence": 1.0,
                            "speaker": current_speaker
                        }]
                    
                    await websocket.send(json.dumps({
                        "channel": {"alternatives": [{
                            "transcript": text, 
                            "confidence": 1.0,
                            "speaker": current_speaker, # Added top-level consistency
                            "words": words_forced
                        }]},
                        "is_final": True
                    }, ensure_ascii=False))
                    
                    current_speaker = 1 - current_speaker # Toggle speaker
                 
                 # [CRITICAL FIX] Always clear buffer after Forced Segmentation, even if text is empty!
                 rolling_buffer = []

                 current_sentence_id += 1 # New sequence 
                 client_vad.reset()
                 
                 # [TIMESTAMP FIX] Update VAD Offset because reset() zeroes internal counter
                 vad_start_offset_samples = total_samples_processed
            
            # B. PARTIAL DECODE (Visual Feedback)
            # [OPTIMIZATION] Dynamic Interval to prevent locking on long sentences
            # Base = DECODE_INTERVAL (e.g. 0.2s or 0.4s)
            # Add 0.05s for every second of audio. 10s audio -> +0.5s interval.
            buffer_duration = len(rolling_buffer) / 16000.0
            dynamic_interval = max(DECODE_INTERVAL, DECODE_INTERVAL + buffer_duration * 0.05)
            
            if len(rolling_buffer) > 4000 and (current_time - last_decode_time > dynamic_interval):
                if not is_partial_decoding:
                    # [SAFE ASYNC] Fire-and-forget task
                    # Copy buffer to ensure thread safety
                    buffer_copy = np.array(rolling_buffer, dtype=np.float32)
                    is_partial_decoding = True
                    # Pass sequence ID to ensure we don't send stale partials after final
                    asyncio.create_task(run_partial_decode(buffer_copy, current_sentence_id))
                else:
                    # Previous decode still running, skip this frame to prevent stacking
                    pass

    except websockets.exceptions.ConnectionClosed:
        logging.info("🔌 Client đã ngắt kết nối")
    except Exception as e:
        logging.error(f"❌ Lỗi connection: {e}")

# Helper for threaded decoding
def decode_buffer_sync(recognizer, buffer_array):
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, buffer_array)
    recognizer.decode_stream(stream)
    return stream.result.text.strip().lower()

async def main():
    server = await websockets.serve(handle_connection, "0.0.0.0", PORT, ping_interval=None)
    logging.info(f"🚀 Server UPDATED VERSION (Async Partial Decode) đang lắng nghe tại ws://0.0.0.0:{PORT}")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())