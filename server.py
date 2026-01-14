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
        num_threads=4,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
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
    model_dir = "./model_vi"
    vad_model = os.path.join(model_dir, "silero_vad.onnx")
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = vad_model
    vad_config.sample_rate = 16000
    
    # ⚡ TUNING VAD PARAMETERS (SENSITIVITY BOOST)
    # Giảm threshold xuống 0.3 để bắt giọng nói nhỏ/xa tốt hơn
    # ⚡ TUNING VAD PARAMETERS
    # Tang threshold len 0.45 de do bi noise lam treo cau
    vad_config.silero_vad.threshold = 0.45         
    vad_config.silero_vad.min_silence_duration = 0.5
    vad_config.silero_vad.min_speech_duration = 0.25 
    
    client_vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)
    
    # --- PSEUDO-STREAMING LOGIC ---
    rolling_buffer = [] 
    last_decode_time = 0
    DECODE_INTERVAL = 1.0 # [TUNING] Reduce to 1.0s for better responsiveness
    
    current_sentence_id = 0
    current_speaker = 0 
    last_segment_end_time = 0 # Track time to detect long pauses

    try:
        async for message in websocket:
            # message is bytes (audio chunk)
            samples = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
            
            # [SMART GAIN CONTROl]
            # Thay vì nhân 3 cứng nhắc (dễ vỡ tiếng nếu mic gần), ta dùng cơ chế "Normalize" nhẹ
            # Nếu âm lượng quá bé (max < 0.1), mới boost lên.
            
            max_amp = np.max(np.abs(samples)) if len(samples) > 0 else 0
            
            # [NOISE GATE] Prevent amplifying silence/noise
            # If signal is too weak (< 0.03), treat as silence/noise to ignore.
            if max_amp < 0.03:
                # Still feed to VAD so it can detect silence duration (End of Speech)
                client_vad.accept_waveform(samples)
                # But DO NOT accumulate to buffers (prevents Forced Segmentation on noise)
                # And DO NOT boost gain (amplifying noise causes "Ừ" "Ờ" output)
                pass 
            else:
                if max_amp > 0 and max_amp < 0.3: # Chỉ boost nếu âm thanh thực sự nhỏ
                    target_gain = 0.5 / max_amp # Target mức 0.5 (an toàn)
                    # Giới hạn Gain không quá 5 lần để tránh noise floor bị rồ lên
                    perform_gain = min(target_gain, 4.0) 
                    samples = samples * perform_gain
                
                # Clip an toàn
                samples = np.clip(samples, -1.0, 1.0)
                
                # 1. Add to rolling buffer (for partial results)
                rolling_buffer.extend(samples)
                
                # 2. Feed to VAD (for final decision)
                client_vad.accept_waveform(samples)
            
            # [DEBUG] Track buffer size to ensure it's growing
            if len(rolling_buffer) % 16000 < len(samples): 
                # Log roughly every second
                # logging.info(f"📊 Buffer Size: {len(rolling_buffer)} / 96000")
                pass

            current_time = asyncio.get_event_loop().time()
            
            # A. CHECK FOR FINAL SEGMENTS (VAD Decision)
            # If VAD has found a segment, it means a sentence has finished.
            while not client_vad.empty():
                speech_segment = client_vad.front.samples
                client_vad.pop()
                
                # [FIX] Removed Manual Prepend History to prevent "Word Repetition" (e.g. "Trời hôm nay")
                # VAD usually handles boundaries well enough.
                
                # New: Calculate offset for timestamps
                segment_offset_seconds = 0.0
                if hasattr(client_vad.front, 'start'):
                     segment_offset_seconds = client_vad.front.start / 16000.0
                
                # Decode the Clean Segment
                stream = recognizer.create_stream()
                stream.accept_waveform(16000, speech_segment)
                recognizer.decode_stream(stream)
                result = stream.result
                
                # Reconstruct text properly
                if hasattr(result, 'tokens'):
                     raw_tokens = result.tokens
                     # [FIX] Lowercase để đồng bộ
                     reconstructed_text = "".join(raw_tokens).replace('▁', ' ').strip().lower()
                     import re
                     # Chỉ để raw text là chữ thường, Frontend tự lo viết hoa
                     text = re.sub(r'\s+', ' ', reconstructed_text)
                else:
                     text = result.text.strip().lower()
                
                if text:
                    # [SMART PARAGRAPHING] Toggle Speaker ONLY on long pause (> 2.0s)
                    time_gap = current_time - last_segment_end_time
                    if last_segment_end_time > 0 and time_gap > 2.0:
                        current_speaker = 1 - current_speaker # New Paragraph
                        logging.info(f"¶ New Paragraph (Gap: {time_gap:.2f}s)")
                        
                    last_segment_end_time = current_time # Update for next time

                    # [FEATURE] Word-level Timestamps & Speaker Toggle
                    words = []
                    if hasattr(result, 'tokens') and hasattr(result, 'timestamps'):
                        for i, token in enumerate(result.tokens):
                             local_start = result.timestamps[i]
                             
                             # Calculate Absolute Timestamp
                             absolute_start = segment_offset_seconds + local_start # Removed prepend_duration
                             absolute_start = max(0.0, absolute_start)
                             
                             start = absolute_start
                             end = start + 0.1
                             
                             if i < len(result.timestamps) - 1:
                                 next_local = result.timestamps[i+1]
                                 next_absolute = segment_offset_seconds + next_local # Removed prepend_duration
                                 end = next_absolute
                             
                             # [FIX] Lower clean_word
                             clean_word = token.replace('▁', '').strip().lower()
                             words.append({
                                 "word": clean_word,
                                 "start": round(start, 2),
                                 "end": round(end, 2),
                                 "confidence": 1.0,
                                 "speaker": current_speaker
                             })
                    
                    # [FALLBACK] Ensure "words" is not empty so client can access words[0].speaker
                    if not words:
                        words.append({
                            "word": text,
                            "start": round(segment_offset_seconds, 2),
                            "end": round(segment_offset_seconds + 1.0, 2),
                            "confidence": 1.0,
                            "speaker": current_speaker
                        })
                    
                    # [FINAL RESULT]
                    logging.info(f"✅ Final Result [Speaker {current_speaker}]: {text}")
                    response = {
                        "channel": {
                            "alternatives": [{
                                "transcript": text,
                                "confidence": 1.0,
                                "speaker": current_speaker, # Top-level speaker field
                                "words": words # Rich metadata
                            }]
                        },
                        "is_final": True
                    }
                    await websocket.send(json.dumps(response, ensure_ascii=False))
                    
                    # [REMOVED] Always Toggle
                    # current_speaker = 1 - current_speaker
                
                # Reset rolling buffer because we justified finished a sentence
                rolling_buffer = []  
                
            # [CRITICAL FEATURE] C. FORCED SEGMENTATION (Prevent Freezing on Long Speech)
            # If user speaks continuously for > 10 seconds without silence, FORCE a cut.
            # Set to 10s as a balanced limit
            if len(rolling_buffer) > 160000: # 16000 * 10s
                 logging.info(f"⚠️ Forced Segmentation (Long Speech > 10s)")
                 stream = recognizer.create_stream()
                 stream.accept_waveform(16000, np.array(rolling_buffer, dtype=np.float32))
                 recognizer.decode_stream(stream)
                 # [FIX] Lowercase forced segment
                 text = stream.result.text.strip().lower()
                 
                 if text:
                    logging.info(f"✅ Final Result (Forced - Speaker {current_speaker}): {text}")
                    
                    # Minimal words construction for forced segment (timestamps harder here)
                    words_forced = [{
                        "word": text,
                        "start": 0.0,
                        "end": 10.0,
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
                 client_vad.reset()
            
            # B. PARTIAL DECODE (Visual Feedback)
            # Only if we have enough data and enough time passed
            if len(rolling_buffer) > 4000 and (current_time - last_decode_time > DECODE_INTERVAL):
                # Convert rolling buffer to numpy for decoding
                # [OPTIMIZATION] Decode FULL buffer (up to 15s max)
                # We removed the slice limit to prevent "text trimming" (disappearing words).
                
                buffer_array = np.array(rolling_buffer, dtype=np.float32)
                
                stream = recognizer.create_stream()
                stream.accept_waveform(16000, buffer_array)
                recognizer.decode_stream(stream)
                text = stream.result.text.strip().lower() # lowercase for partial
                
                if text:
                    # [PARTIAL RESULT]
                    # logging.info(f"Typing... {text}") # Too noisy for logs
                    response = {
                        "channel": {
                            "alternatives": [{
                                "transcript": text,
                                "confidence": 0.5,
                            }]
                        },
                        "is_final": False
                    }
                    await websocket.send(json.dumps(response, ensure_ascii=False))
                
                last_decode_time = current_time

    except websockets.exceptions.ConnectionClosed:
        logging.info("🔌 Client đã ngắt kết nối")
    except Exception as e:
        logging.error(f"❌ Lỗi connection: {e}")

async def main():
    server = await websockets.serve(handle_connection, "0.0.0.0", PORT, ping_interval=None)
    logging.info(f"🚀 Server UPDATED VERSION (Speaker Toggle + Rich Meta) đang lắng nghe tại ws://0.0.0.0:{PORT}")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())